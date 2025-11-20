#include "ops.h"
#include <stdexcept>

Tensor AddOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("AddOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];

  return a+b;
}

Tensor MulOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("MulOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];
  return a*b;
}

Tensor MatmulOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("MatmulOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];
  return a.matmul(b);
}

Tensor add(const Tensor& a, const Tensor& b) {
  AddOp op;
  return op.forward({a, b});
}

Tensor mul(const Tensor& a, const Tensor& b) {
  MulOp op;
  return op.forward({a, b});
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  MatmulOp op;
  return op.forward({a, b});
}