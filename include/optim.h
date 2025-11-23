#pragma once
#include "tensor.h"
#include <vector>

class SGD {
public:
  float lr;
  vector<Tensor*> params;

  // params: model weight & bias
  SGD(const vector<Tensor*>& params, float lr=0.01f):
    params(params), lr(lr) {}
  
  void step();
  void zero_grad();
};