#pragma once 
#include "tensor.h"
#include "module.h"
#include <random>

class Dense: public Module {
public:
  Dense(int in_features, int out_features, bool bias=true, float wscale=0.1f);
  Tensor forward(const Tensor& x) override;
  vector<Tensor*> parameters() override;
  
  Tensor W; // (in_features, out_features)
  Tensor b; // (out_features,)

private: 
  bool use_bias;
};