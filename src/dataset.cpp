#include "dataset.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <filesystem>

static int32_t read_be32(ifstream& f) {
  unsigned char b[4];
  f.read((char*)b, 4);
  return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
}

MNISTDataset::MNISTDataset(const string& root, bool train) {
  string img_path = train ? (root+"/train-images.idx3-ubyte") : (root+"/t10k-images.idx3-ubyte");
  string lab_path = train ? (root+"/train-labels.idx1-ubyte") : (root+"/t10k-labels.idx1-ubyte");
  if (!filesystem::exists(img_path) || !filesystem::exists(lab_path)) 
    throw runtime_error("MNISTDataset: files not found");

  load_images(img_path);
  load_labels(lab_path);
  if (labels.size() != images.size()/(28*28))
    throw runtime_error("MNISTDataset: image/label count mismatch");
}

void MNISTDataset::load_images(const string& path) {
  ifstream f(path, ios::binary);

  int32_t magic = read_be32(f);
  int32_t n = read_be32(f);
  int32_t rows = read_be32(f);
  int32_t cols = read_be32(f);
  if (magic!=2051 || rows!=28 || cols!=28)
    throw runtime_error("invalid MNIST image file");
  images.resize(n*28*28);
  vector<unsigned char> buf(28*28);
  for (int i=0; i<n; i++) {
    f.read((char*)buf.data(), buf.size()); // read 1 image data into <buf>
    // read each char in <buf>
    for (int j=0; j<28*28; j++)
      images[i*28*28+j] = buf[j]/255.0f; // normalized [0..1]
  }
}

void MNISTDataset::load_labels(const string& path) {
  ifstream f(path, ios::binary);
  if (!f) 
    throw runtime_error("failed to open " + path);
  int32_t magic = read_be32(f);
  int32_t n = read_be32(f);
  if (magic != 2049)
    throw runtime_error("invalid MNIST label file");
  labels.resize(n);
  for (int i=0; i<n; i++) {
    unsigned char b;
    f.read((char*)&b, 1); // read into <b> 1 byte
    labels[i] = (int)b;
  }
}

void MNISTDataset::get_item(int idx, Tensor& x, Tensor& y) const {
  x = Tensor({28*28}, vector<float>(
    images.begin() + idx*28*28,
    images.begin() + (idx+1)*28*28
  ));
  y = Tensor({(float)labels[idx]});
}

// ---------------- DataLoader ---------------
DataLoader::DataLoader(const Dataset& dataset_, int batch_size_, bool shuffle):
  dataset(dataset_), batch_size(batch_size_), shuffle_flag(shuffle) {
  int n = dataset.size();
  indices.resize(n);
  for (int i=0; i<n; i++)
    indices[i] = i;
  
  if (shuffle_flag) 
    std::shuffle(indices.begin(), indices.end(), mt19937{123});
}

void DataLoader::reset() {
  cursor = 0;
  if (shuffle_flag)
    std::shuffle(indices.begin(), indices.end(), mt19937{std::random_device{}()});
}

// return false when epoch finishes
bool DataLoader::next_batch(Tensor& X, Tensor& Y) {
  if (cursor >= (int)indices.size())
    return false;
  
  int end = std::min(cursor+batch_size, (int)indices.size());
  int b = end - cursor; // real batch size

  X = Tensor(b, 28*28); // (batch,dim)
  Y = Tensor(b); // 1D tensor of <b> elements

  for (int i=0; i<b; i++) {
    Tensor xi, yi; // 1 image, 1 label
    dataset.get_item(indices[cursor+i], xi, yi);
    // loop over each pixel in 1 image
    for (int j=0; j<28*28; j++)
      X.at(i,j) = xi[j];
    Y[i] = yi[0];
  }
  cursor = end;
  return true;
}