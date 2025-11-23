#pragma once 
#include "tensor.h"
#include "ops.h"
#include <vector>

class Linear {
public:
  Tensor W; // (in_features, out_features)
  Tensor b; // (out_features,)

  Linear(int in_features, int out_features);
  
  Tensor operator()(const Tensor& x); // forward
};