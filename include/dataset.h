#pragma once 
#include <vector>
#include <string>
#include <random>
#include "tensor.h"

class Dataset {
public:
  virtual ~Dataset() = default;
  virtual int size() const = 0;
  virtual void get_item(int idx, Tensor& x, Tensor& y) const = 0;
};

// root = folder containing:
//  train-images.idx3-ubyte
//  train-labels.idx1-ubyte
//  t10k-images.idx3-ubyte
//  t10k-labels.idx1-ubyte
class MNISTDataset: public Dataset {
public:
  MNISTDataset(const string& root, bool train=true);

  // number of images
  int size() const override {
    return images.size() / (28*28);
  }

  void get_item(int idx, Tensor& x, Tensor& y) const override;

private:
  vector<float> images; // normalized 0..1
  vector<int> labels;

  void load_images(const string& path);
  void load_labels(const string& path);
};

class DataLoader {
public:
  DataLoader(const Dataset& dataset, int batch_size, bool shuffle=true);
  // return false when epoch finishes
  bool next_batch(Tensor& X, Tensor& Y);
  void reset(); // begin new epoch

private:
  const Dataset& dataset;
  int batch_size;
  bool shuffle_flag;
  vector<int> indices;
  int cursor = 0;
};