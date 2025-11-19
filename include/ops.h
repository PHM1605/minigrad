#pragma once 
#include <vector>

class Tensor; // forward declaration

// base class for all operations
struct Op {
  virtual ~Op() = default;
  // e.g. c = add(a,b) then inputs = ref to vector of pointers to <a>,<b> 
  virtual Tensor forward(const std::vector<Tensor*>& inputs) = 0;
};