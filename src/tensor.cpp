#include "tensor.h"
#include <iostream>
#include <cassert>
#include <iomanip> // for format in Tensor::print()

// Constructors
Tensor::Tensor(): shape({0}), data() {}
Tensor::Tensor(int n): shape({n}), data(n, 0.0f) {}
Tensor::Tensor(int rows, int cols): shape({rows, cols}), data(rows*cols, 0.0f) {}

Tensor::Tensor(const vector<int>& shape_, float fill):
  shape(shape_) {
  int n = prod(shape); // number of elements in Tensor
  data.assign(n, fill);
}

Tensor::Tensor(const vector<int>& shape_, const vector<float>& values)
: shape(shape_) {
  int n = prod(shape);
  if ((int)values.size() != n) throw runtime_error("Tensor: values size != shape product");
  data = values;
}

Tensor::Tensor(initializer_list<float> list):
  shape({(int)list.size()}), data(list) {}

Tensor::Tensor(initializer_list<initializer_list<float>> mat) {
  int r = (int)mat.size();
  int c = r ? (int)mat.begin()->size() : 0;
  shape = {r, c};
  data.reserve(r*c);
  for (auto &row: mat) {
    // if any row length != 1st row length
    if ((int)row.size() != c) 
      throw runtime_error("Tensor: ragged initializer not allowed");
    for (float v: row)
      data.push_back(v);
  } 
}

int Tensor::ndim() const {
  return (int)shape.size();
}

// total number of elements in Tensor 
int Tensor::size() const {
  return prod(shape);
}

int Tensor::rows() const {
  if (shape.size() != 2)
    throw runtime_error("Tensor::rows() requires 2D tensor");
  return shape[0];
}

int Tensor::cols() const {
  if (shape.size() != 2)
    throw runtime_error("Tensor::cols() requires 2D tensor");
}

// reshape; but only change <shape> property for now
void Tensor::reshape(const std::vector<int>& new_shape) {
  if (prod(new_shape) != size())
    throw runtime_error("Tensor::reshape() total size mismatch");
  shape = new_shape;
}

string Tensor::shape_str() const {
  string s = "(";
  for (size_t i=0; i<shape.size(); ++i) {
    s += to_string(shape[i]);
    if (i+1 < shape.size())
      s += ",";
  }
  s += ")";
  return s;
}

// for non-const 2D version
float& Tensor::at(int r, int c) {
  int R = rows(), C = cols();
  if (r<0 || r>=R || c<0 || c>=C)
    throw out_of_range("Tensor::at index");
  return data[r*C+c];
} 
// for const 2D version
float Tensor::at(int r, int c) const {
  int R = rows(), C = cols();
  if (r<0 || r>=R || c<0 || c>=C)
    throw out_of_range("Tensor::at index");
  return data[r*C+c];
}

// for non-const 1D version
float& Tensor::operator[](int i) {
  if (i<0 || i>=(int)data.size())
    throw out_of_range("Tensor::operator[]");
  return data[i];
}
// for const 1D version
float Tensor::operator[](int i) const {
  if (i<0 || i>=(int)data.size())
    throw out_of_range("Tensor::operator[]");
}

Tensor Tensor::operator+(const Tensor& other) const {
  assert(size == other.size);
  Tensor out(size);
  for (int i=0; i<size; i++)
    out.data[i] = data[i] + other.data[i];
  return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
  assert(size == other.size);
  Tensor out(size);
  for (int i=0; i<size; i++) {
    out.data[i] = data[i] * other.data[i];
  }
  return out;
}

// XOR: treats non-zero as True
Tensor Tensor::logical_xor(const Tensor& other) const {
  assert(size == other.size);
  Tensor out(size);
  for (int i=0; i<size; ++i) {
    bool a = (data[i] != 0.0f); // convert any value to 0 or 1
    bool b = (other.data[i] != 0.0f); // convert any value to 0 or 1
    out.data[i] = (a^b) ? 1.0f : 0.0f;
  }
  return out;
}

void Tensor::print() const {
  for(float v: data) {
    cout << v << " ";
  }
  cout << "\n";
}

// fill the Tensor with this value
void Tensor::fill(float v) {
  std::fill(data.begin(), data.end(), v);
}

// product of all elements in <v>
int Tensor::prod(const std::vector<int>& v) {
  if (v.empty())
    return 0;
  int p = 1;
  for (int x: v)
    p *= x;
  return p;
}
