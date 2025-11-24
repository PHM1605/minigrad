#pragma once 
#include "tensor.h"
#include "layers.h"
#include "ops.h"
#include <vector>
#include <memory>

class MLP {
public:
  vector<Linear> layers;
  vector<string> activations;
  
  MLP(const vector<int>& sizes, const vector<string>& activs); // "relu", "sigmoid", "tanh", "none"

  Tensor operator() (const Tensor& x);
  vector<Tensor*> parameters();
};