#include "ops.h"
#include <stdexcept>
#include <iostream>

// compute broadcasted shape
vector<int> broadcast_shape(const vector<int>& a, const vector<int>& b) {
  // take max number of dimensions
  int na = a.size();
  int nb = b.size();
  int n = max(na, nb);
  vector<int> out(n);

  for (int i=0; i<n; i++) {
    int ai = (i<n-na) ? 1 : a[i-(n-na)];
    int bi = (i<n-nb) ? 1 : b[i-(n-nb)];
    if (ai != bi && ai!=1 && bi!=1)
      throw runtime_error("Incompatible for broadcasting");
    out[i] = max(ai, bi);
  }
  return out;
}

// reduce grad from (5,3)->(3,)
// because (5,3) is only the result of broadcasting true input (3,) 
static Tensor reduce_sum_to_shape(const Tensor& grad, const vector<int>& target_shape) {
  if (grad.shape() == target_shape) 
    return grad;
  
  const auto& gshape = grad.shape();
  int grad_dim = (int)gshape.size();
  int true_dim = (int)target_shape.size();
  if (true_dim > grad_dim)
    throw runtime_error("reduce_sum_to_shape: target has higher rank than grad");

  // padded_target_shape: [1,3]
  vector<int> padded_target_shape(grad_dim, 1); 
  for (int i=0; i<true_dim; i++) {
    padded_target_shape[i+grad_dim-true_dim] = target_shape[i];
  }

  // prepare output (3,)
  Tensor out(target_shape);
  out.fill(0.0f);
  int out_total = out.size();
  int grad_total = grad.size();

  // stride of each dimension of <grad>
  vector<int> grad_stride(grad_dim, 1); 
  // modify stride of next-to-last till beginning
  for (int d=grad_dim-2; d>=0; --d) 
    grad_stride[d] = grad_stride[d+1]*gshape[d+1];

  // stride of each dimension of <true>
  vector<int> true_stride(true_dim, 1);
  for (int d=true_dim-2; d>=0; --d)
    true_stride[d] = true_stride[d+1]*target_shape[d+1];

  // flatting grad tensor
  for (int grad_flat=0; grad_flat < grad_total; ++grad_flat) {
    // convert grad-flat-index to multi-index
    int tmp = grad_flat;
    // this <flat> cell belongs to which indices
    vector<int> grad_indices(grad_dim);
    for (int d=0; d<grad_dim; ++d) {
      grad_indices[d] = tmp / grad_stride[d];
      tmp %= grad_stride[d];
    }

    // grad multi-index => true multi-index
    vector<int> true_indices(true_dim, 0);
    for (int d=0; d<true_dim; d++) {
      // gd: axis <d> of <true_dim> is which axis in <grad_dim>
      int gd = d+(grad_dim-true_dim);
      if (padded_target_shape[gd] != 1)
        true_indices[d] = grad_indices[gd];
    }

    // true multi-index => flat index
    int out_flat_idx = 0;
    for (int d=0; d<true_dim; d++)
      out_flat_idx += true_indices[d]* true_stride[d];
    
    // sum all broadcasted gradients
    out[out_flat_idx] += grad.data()[grad_flat];
  }

  return out; // (3,)
}

Tensor AddOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("AddOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];

  vector<int> out_shape = broadcast_shape(a.shape(), b.shape());

  Tensor A = (a.shape() == out_shape) ? a : a.broadcast_to(out_shape);
  Tensor B = (b.shape() == out_shape) ? b : b.broadcast_to(out_shape);

  return A+B;
}

vector<Tensor> AddOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {
  if (inputs.size() != 2) {
    throw runtime_error("AddOp backward expects 2 inputs");
  }
  // a+b=c => dL/da = dL/dc*dc/da = out_grad*1; dL/db = dL/dc*dc/db = out_grad*1
  Tensor ga = out_grad; // (5,3)
  Tensor gb = out_grad;  // (5,3)

  // reduce to match input shapes if they were broadcasted
  vector<int> ashape = inputs[0].shape(); // (3,)
  vector<int> bshape = inputs[1].shape(); // (5,3)
  if (ga.shape() != ashape)
    ga = reduce_sum_to_shape(ga, ashape); // (5,3)=>(3,)
  if (gb.shape() != bshape)
    gb = reduce_sum_to_shape(gb, bshape); // (5,3)=>(3,);

  return {ga, gb};
}

Tensor SubOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("SubOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];

  vector<int> out_shape = broadcast_shape(a.shape(), b.shape());

  Tensor A = (a.shape() == out_shape) ? a : a.broadcast_to(out_shape);
  Tensor B = (b.shape() == out_shape) ? b : b.broadcast_to(out_shape);

  return A-B;
}

vector<Tensor> SubOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {
  if (inputs.size() != 2) {
    throw runtime_error("SubOp backward expects 2 inputs");
  }
  // a-b=c => dL/da = dL/dc*dc/da = out_grad*1; dL/db = dL/dc*dc/db = out_grad*(-1)
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];

  Tensor ga = out_grad; // (5,3)
  Tensor neg_one({-1.0f});
  Tensor neg = neg_one.broadcast_to(out_grad.shape());
  Tensor gb = mul(out_grad, neg);  // (5,3)

  // reduce to match input shapes if they were broadcasted
  if (ga.shape() != a.shape())
    ga = reduce_sum_to_shape(ga, a.shape()); // (5,3)=>(3,)
  if (gb.shape() != b.shape())
    gb = reduce_sum_to_shape(gb, b.shape()); // (5,3)=>(3,);

  return {ga, gb};
}

Tensor MulOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("MulOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];

  vector<int> out_shape = broadcast_shape(a.shape(), b.shape());

  Tensor A = (a.shape() == out_shape) ? a : a.broadcast_to(out_shape);
  Tensor B = (b.shape() == out_shape) ? b : b.broadcast_to(out_shape);

  return A*B;
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

  // reduce to original shapes if necessary
  if (grad_a.shape() != a.shape())
    grad_a = reduce_sum_to_shape(grad_a, a.shape());
  if (grad_b.shape() != b.shape())
    grad_b = reduce_sum_to_shape(grad_b, b.shape());
    
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
  if (inputs.size() != 1) {
    throw runtime_error("ReduceSumOp expects 1 input");
  }
  const Tensor& x = inputs[0];
  float s = 0.0f;
  for (float v: x.data())
    s += v;
  Tensor out({s}); // Tensor (1,)

  return out;
}

vector<Tensor> ReduceSumOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {
  if (inputs.size() != 1) {
    throw runtime_error("ReduceSumOp backward expects 1 input");
  }
  const Tensor& x = inputs[0];
  if (out_grad.size() != 1) {
    throw runtime_error("SumOp backward: out_grad must be scalar");
  }
  // dL/dx = [dL/dx0, dL/dx1, dL/dx2]; each dL/dxi=dL/dy*dy/dxi=out_grad*1
  Tensor gx(x.shape(), 0.0f); // output tensor
  float g = out_grad.data()[0]; 
  for (int i=0; i<gx.size(); ++i) {
    gx[i] = g;
  }
  return {gx};
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

Tensor sub(const Tensor& a, const Tensor& b) {
  auto op = make_shared<SubOp>();
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
  return out;
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