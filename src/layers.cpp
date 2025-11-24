#include "layers.h"
#include "ops.h"
#include <chrono>

Dense::Dense(int in_features, int out_features, bool bias, float wscale):
  W(vector<int>{in_features, out_features}, 0.0f),
  b(vector<int>{out_features}, 0.0f),
  use_bias(bias)
{
  // Random init
  mt19937 rng(123);
  uniform_real_distribution<float> dist(-0.1f, 0.1f);
  for (int i=0; i<W.size(); ++i) {
    W[i] = dist(rng);
  } 
  if (use_bias) {
    // Init bias to 0
    for (int i=0; i<b.size(); ++i) {
      b[i] = 0.0f;
    }
  }
  W.set_requires_grad(true);
  if (use_bias)
    b.set_requires_grad(true);
}

Tensor Dense::forward(const Tensor& x) {
  // x: (batch, in_features) expected
  // W: (in_features, out_features)
  Tensor y = matmul(x, W); // (batch, out_features)
  if (use_bias) {
    Tensor bb = broadcast(b, y.shape()); // (batch, out_features)
    y = add(y, bb);
  }
  return y;
}

vector<Tensor*> Dense::parameters() {
  vector<Tensor*> p;
  p.push_back(&W);
  if (use_bias) 
    p.push_back(&b);
  return p;
}