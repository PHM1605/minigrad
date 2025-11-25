#include "ops.h"
#include <stdexcept>
#include <iostream>
#include <math.h>
#include <cmath>

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

Tensor BroadcastOp::forward(const vector<Tensor>& inputs) const {
  return inputs[0].broadcast_to(target_shape);
}

vector<Tensor> BroadcastOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& /*output*/) const {
  Tensor g = out_grad;
  if (g.shape() != inputs[0].shape())
    g = reduce_sum_to_shape(g, inputs[0].shape());
  return {g};
}

Tensor AddOp::forward(const vector<Tensor>& inputs) const {
  if (inputs.size() != 2) {
    throw runtime_error("AddOp expects 2 inputs");
  }
  const Tensor& a = inputs[0];
  const Tensor& b = inputs[1];

  vector<int> out_shape = broadcast_shape(a.shape(), b.shape());

  Tensor A = (a.shape() == out_shape) ? a : broadcast(a, out_shape);
  Tensor B = (b.shape() == out_shape) ? b : broadcast(b, out_shape);

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

  Tensor A = (a.shape() == out_shape) ? a : broadcast(a, out_shape);
  Tensor B = (b.shape() == out_shape) ? b : broadcast(b, out_shape);

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
  Tensor neg = (out_grad.shape() == neg_one.shape())
    ? neg_one
    : broadcast(neg_one, out_grad.shape());
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

  Tensor A = (a.shape() == out_shape) ? a : broadcast(a, out_shape);
  Tensor B = (b.shape() == out_shape) ? b : broadcast(b, out_shape);

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

Tensor ReLUOp::forward(const vector<Tensor>& inputs) const {
  const Tensor& x = inputs[0];
  Tensor out(x.shape(), 0.0f);
  for (int i=0; i<x.size(); i++) {
    out[i] = (x.data()[i] > 0.0f) ? x.data()[i] : 0.0f;
  }
  return out;
}

vector<Tensor> ReLUOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& output) const {
  const Tensor& x = inputs[0];
  Tensor gx(x.shape(), 0.0f);
  for (int i=0; i<x.size(); i++) {
    gx[i] = (x.data()[i] > 0.0f) ? out_grad.data()[i] : 0.0f;
  }
  return {gx};
}

// Sigmoid: y = 1/(1+exp(-x))
Tensor SigmoidOp::forward(const vector<Tensor>& inputs) const {
  const Tensor& x = inputs[0];
  Tensor out(x.shape(), 0.0f);
  for (int i=0; i<x.size(); i++) {
    out[i] = 1.0f / (1.0f + exp(-x.data()[i]));
  }
  return out;
}

// dL/dx = dL/dy*dy/dx = out_grad*( y*(1-y) )
vector<Tensor> SigmoidOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& output) const {
  Tensor grad_x(output.shape(), 0.0f);
  for (int i=0; i<output.size(); i++) {
    float y = output.data()[i];
    grad_x[i] = out_grad.data()[i]*y*(1-y);
  }
  return {grad_x};
}

Tensor TanhOp::forward(const vector<Tensor>& inputs) const {
  const Tensor& x = inputs[0];
  Tensor out(x.shape(), 0.0f);
  for (int i=0; i<x.size(); i++) {
    out[i] = tanh(x.data()[i]);
  }
  return out;
}

// dL/dx = dL/dy*dy/dx = out_grad*(1-y^2)
vector<Tensor> TanhOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& output) const {
  Tensor grad_x(output.shape(), 0.0f);
  for (int i=0; i<output.size(); i++) {
    float y = output.data()[i];
    grad_x[i] = out_grad.data()[i]*(1-y*y);
  }
  return {grad_x};
}

// p_i = exp(z_i) / sum_over_j(z_j)
Tensor SoftmaxOp::forward(const vector<Tensor>& inputs) const {
  const Tensor& logits = inputs[0]; // (batch,dim)
  int R = logits.rows();
  int C = logits.cols();
  Tensor out({R,C}, 0.0f);
  for (int r=0; r<R; r++) {
    float maxv = -1e30f;
    for (int c=0; c<C; c++)
      maxv = max(maxv, logits.at(r,c));
    // NOTE: we use here the trick softmax(x) = softmax(x-const) to prevent e^too_big = +inf
    float sum = 0;
    for (int c=0; c<C; c++) {
      float e = exp(logits.at(r,c) - maxv);
      out.at(r,c) = e;
      sum += e;
    }
    // normalize
    for (int c=0; c<C; c++)
      out.at(r,c) /= sum;
  }
  return out;
}

vector<Tensor> SoftmaxOp::backward(const Tensor& grad_out, const vector<Tensor>& inputs, const Tensor& output) const {
  const Tensor& y = output;
  Tensor grad_x(y.shape(), 0.0f);
  int R = y.rows();
  int C = y.cols();
  for (int r=0; r<R; r++) {
    // check one row first; dL/dz_i = sum_over_j( dL/dy_j*dy_j/dz_i )
    // then dy_j/dz_i = y_i*(kronecker_delta_ij - y_j) 
    // with kronecker_delta_ij=1 when i==j else 0
    for (int i=0; i<C; i++) {
      float sum=0;
      for (int j=0; j<C; j++) {
        float delta = (i==j) ? 1.0f : 0.0f;
        sum += grad_out.at(r,j) * y.at(r,i)*(delta-y.at(r,j));
      }
      grad_x.at(r,i) = sum;
    }
  }
  return {grad_x};
}

// loss = -log( softmax(logits)[target] ) = -log( exp(logits_true)/SUM ) = -(logits_true-log(SUM)) 
Tensor CrossEntropyOp::forward(const vector<Tensor>& inputs) const {
  const Tensor& logits = inputs[0]; // (batch,dim)
  const Tensor& targets = inputs[1]; // (batch,)
  int R = logits.rows(); // batch
  int C = logits.cols(); // dim
  Tensor loss({R,1}, 0.0f);
  
  for (int r=0; r<R; r++) {
    // find the max logit_value in each row
    float maxv = -1e30;
    for (int c=0; c<C; c++)
      maxv = max(maxv, logits.at(r,c));
    // SUM of exp of logits
    float sum = 0;
    for (int c=0; c<C; c++)
      sum += exp(logits.at(r,c)-maxv);

    int t = (int)targets[r]; // true class index
    float logp = logits.at(r,t)-maxv - log(sum); // logits_true-log(SUM)

    loss.at(r,0) = -logp;
  }

  return loss;
}

// L = log(SUM) - z_true
// dlog(SUM)/dz_i = softmax(z_i)
// => if i==true then dL/dz_i = softmax(z_i)-1
// => if i!=true then dL/dz_i = softmax(z_i)
vector<Tensor> CrossEntropyOp::backward(const Tensor& out_grad, const vector<Tensor>& inputs, const Tensor& output) const {
  const Tensor& logits = inputs[0]; // (batch,dim)
  const Tensor& targets = inputs[1]; // (batch,)
  int R = logits.rows();
  int C = logits.cols();
  Tensor soft = softmax(logits);
  Tensor grad_x({R,C}, 0.0f);

  for (int r=0; r<R; r++) {
    int t = (int)targets[r];
    for (int c=0; c<C; c++) {
      float  grad = soft.at(r,c);
      if (c==t)
        grad -= 1.0f;
      grad_x.at(r,c) = grad;
    }
  }
  return {grad_x, Tensor({0.0f})}; // no grad for targets; we won't use it anyway
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

Tensor broadcast(const Tensor& x, const vector<int>& new_shape) {
  auto op = make_shared<BroadcastOp>(new_shape);
  Tensor out = op->forward({x});
  if (x.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {x});
  }
  return out;
}

Tensor relu(const Tensor& x) {
  auto op = make_shared<ReLUOp>();
  Tensor out = op->forward({x});
  if (x.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {x});
  }
  return out;
}

Tensor sigmoid(const Tensor& x) {
  auto op = make_shared<SigmoidOp>();
  Tensor out = op->forward({x});
  if (x.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {x});
  } 
  return out;
}

Tensor tanh_fn(const Tensor& x) {
  auto op = make_shared<TanhOp>();
  Tensor out = op->forward({x});
  if (x.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {x});
  }
  return out;
}

Tensor softmax(const Tensor& logits) {
  auto op = make_shared<SoftmaxOp>();
  Tensor out = op->forward({logits});
  if (logits.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {logits});
  }
  return out;
}

Tensor cross_entropy(const Tensor& logits, const Tensor& targets) {
  auto op = make_shared<CrossEntropyOp>();
  Tensor out = op->forward({logits, targets});
  if (logits.requires_grad()) {
    out.set_requires_grad(true);
    out._set_grad_fn(op, {logits, targets});
  }
  return out;
}