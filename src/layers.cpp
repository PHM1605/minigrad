#include "layers.h"
#include <random>
#include <iostream>

Linear::Linear(int in_features, int out_features) {
  W = Tensor({in_features, out_features}, vector<float>(in_features*out_features, 0.0f));
  b = Tensor({out_features}, vector<float>(out_features, 0.0f));
  // Random init
  mt19937 rng(123);
  uniform_real_distribution<float> dist(-0.1f, 0.1f);
  for (int i=0; i<W.size(); ++i) {
    W[i] = dist(rng);
  } 
  // Init bias to 0
  for (int i=0; i<b.size(); ++i) {
    b[i] = 0.0f;
  }
  W.set_requires_grad(true);
  b.set_requires_grad(true);
}

Tensor Linear::operator()(const Tensor& x) {
  // x: (batch, in_features)
  // W: (in_features, out_features)
  // b: (out_features)
  Tensor y = matmul(x, W); // (batch, out_features)
  return add(y, b);
}