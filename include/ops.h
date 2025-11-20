#pragma once 
#include <vector>
#include <string>
#include <memory>
#include "tensor.h"

using namespace std;

// base class for all operations
struct Op {
  virtual ~Op() = default;
  virtual string name() const = 0;
  // e.g. c = add(a,b) then inputs = ref to vector of pointers to <a>,<b> 
  virtual Tensor forward(const std::vector<Tensor>& inputs) const = 0;
};

class AddOp: public Op {
public: 
  string name() const override {
    return "Add";
  }

  Tensor forward(const vector<Tensor>& inputs) const override;
};

class MulOp: public Op {
public:
  string name() const override {
    return "Mul";
  }
  Tensor forward(const vector<Tensor>& inputs) const override;
};

class MatmulOp: public Op {
public:
  string name() const override {
    return "Matmul";
  }
  Tensor forward(const vector<Tensor>& inputs) const override;
};

Tensor add(const Tensor& a, const Tensor& b);
Tensor mul(const Tensor& a, const Tensor& b);
Tensor matmul(const Tensor& a, const Tensor& b);