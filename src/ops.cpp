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

vector<Tensor> AddOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {
  if (inputs.size() != 2) {
    throw runtime_error("AddOp backward expects 2 inputs");
  }
  // a+b=c => dL/da = dL/dc*dc/da = out_grad*1; dL/db = dL/dc*dc/db = out_grad*1  
  return {out_grad, out_grad};
}

Tensor MulOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("MulOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];
  return a*b;
}

vector<Tensor> MulOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {
  if (inputs.size() != 2) {
    throw runtime_error("MulOp backward expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];
  // z = a*b
  // => grad_a = dL/da = dL/dz*dz/da = out_grad*b
  // => grad_b = dL/db = dL/dz*dz/db = out_grad*a
  Tensor grad_a = out_grad*b;
  Tensor grad_b = out_grad*a;
  return {grad_a, grad_b};
}

Tensor MatmulOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("MatmulOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];
  return a.matmul(b);
}

vector<Tensor> MatmulOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {
  if (inputs.size() != 2) {
    throw runtime_error("MatmulOp backward expects 2 inputs");
  }
  const Tensor& A = inputs[0];
  const Tensor& B = inputs[1];
  // C = A @ B
  // dL/dA = dL/dC @ dC/dA = out_grad @ B^T
  // dL/dB = (dC/dB)^T @ dL/dC = A^T @ out_grad
  Tensor grad_A = out_grad.matmul(B.transpose());
  Tensor grad_B = A.transpose().matmul(out_grad);
  return {grad_A, grad_B};
}

Tensor ReduceSumOp::forward(const vector<Tensor>& inputs) const {

}

vector<Tensor> ReduceSumOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {

}

Tensor add(const Tensor& a, const Tensor& b) {
  auto op = make_shared<AddOp>();
  Tensor out = op->forward({a, b});
  if (a.requires_grad() || b.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {a, b});
  }
  return out;
}

Tensor mul(const Tensor& a, const Tensor& b) {
  auto op = make_shared<MulOp>();
  Tensor out = op->forward({a, b});
  if (a.requires_grad() || b.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {a, b});
  }
  return out;
}

Tensor matmul(const Tensor& a, const Tensor& b) {
  auto op = make_shared<MatmulOp>();
  Tensor out = op->forward({a,b});
  if (a.requires_grad() || b.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {a,b});
  }
}

Tensor reduce_sum(const Tensor& x) {
  auto op = make_shared<ReduceSumOp>();
  Tensor out = op->forward({x});
  if (x.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {x});
  }
  return out;
}