#include "tensor.h"
#include <iostream>
#include <cassert>

Tensor::Tensor(int size): size(size), data(size, 0.0f) {}
Tensor::Tensor(const vector<float>& v): size(v.size()), data(v) {}

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