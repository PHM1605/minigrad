#include "tensor.h"
#include "ops.h"
#include <iostream>
#include <cassert>
#include <iomanip> // for format in Tensor::print()

// product of all elements in <v>
int Tensor::prod(const std::vector<int>& v) {
  int p = 1;
  for (int x: v)
    p *= x;
  return p;
}

// Constructors
Tensor::Tensor() {
  p->realized = true;
}

Tensor::Tensor(int n) {
  p->shape = {n};
  p->data.assign(n, 0.0f);
  p->realized = true;
}

Tensor::Tensor(int rows, int cols) {
  p->shape = {rows, cols};
  p->data.assign(rows*cols, 0.0f);
  p->realized = true;
}

Tensor::Tensor(const vector<int>& shape_, float fill) {
  p->shape = shape_;
  int n = prod(shape_); // number of elements in Tensor
  p->data.assign(n, fill);
  p->realized = true;
}

Tensor::Tensor(const vector<int>& shape_, const vector<float>& values) {
  p->shape = shape_;
  int n = prod(shape_);
  if ((int)values.size() != n) 
    throw runtime_error("Tensor: values size != shape product");
  p->data = values;
  p->realized = true;
}

Tensor::Tensor(initializer_list<float> list) {
  p->shape = {(int)list.size()};
  p->data = list;
  p->realized = true;
}

Tensor::Tensor(initializer_list<initializer_list<float>> mat) {
  int r = (int)mat.size();
  int c = r ? (int)mat.begin()->size() : 0;
  p->shape = {r, c};
  for (auto &row: mat) {
    // if any row length != 1st row length
    if ((int)row.size() != c) 
      throw runtime_error("Tensor: ragged initializer not allowed");
    for (float v: row)
      p->data.push_back(v);
  }
  p->realized = true; 
}

int Tensor::ndim() const {
  return (int)p->shape.size();
}

// total number of elements in Tensor 
int Tensor::size() const {
  return prod(p->shape);
}

int Tensor::rows() const {
  if (p->shape.size() != 2)
    throw runtime_error("Tensor::rows() requires 2D tensor");
  return p->shape[0];
}

int Tensor::cols() const {
  if (p->shape.size() != 2)
    throw runtime_error("Tensor::cols() requires 2D tensor");
  return p->shape[1];
}

vector<int> Tensor::shape() const {
  return p->shape;
}

string Tensor::shape_str() const {
  string s = "(";
  for (size_t i=0; i<p->shape.size(); ++i) {
    s += to_string(p->shape[i]);
    if (i+1 < p->shape.size())
      s += ",";
  }
  s += ")";
  return s;
}

// Lazy realization
bool Tensor::is_realized() const {
  return p->realized;
}

void Tensor::realize() const {
  if (p->realized) return;

  // realize parents first
  for (auto& par: p->parents) {
    par.realize();
  }
  // realizing
  auto op = p->grad_fn;
  Tensor out = op->forward(p->parents);
  
  p->shape = out.p->shape;
  p->data = out.p->data;
  p->realized = true;
}

// Data access
const std::vector<float>& Tensor::data() const {
  realize();
  return p->data;
}
// for non-const 2D version
float& Tensor::at(int r, int c) {
  realize();
  return p->data[r*cols()+c];
} 
// for const 2D version
float Tensor::at(int r, int c) const {
  realize();
  return p->data[r*cols()+c];
}
// for non-const 1D version
float& Tensor::operator[](int i) {
  realize();
  return p->data[i];
}
// for const 1D version
float Tensor::operator[](int i) const {
  realize();
  return p->data[i];
}

void Tensor::print() const {
  realize();
  if (ndim() == 1) {
    for (int i=0; i<size(); i++) {
      cout << p->data[i] << " ";
    }
    cout << "\n";
    return;
  }
  else if (ndim() == 2) {
    int R = rows(), C = cols();
    for (int r=0; r < R; r++) {
      for (int c=0; c<C; c++) {
        cout << setw(6) << at(r,c) << " ";
      }
      cout << "\n";
    }
  } else {
    for (int i=0; i<size(); ++i) 
      cout << p->data[i] << " ";
    cout << "\n";
  }
}

// Basic ops
Tensor Tensor::operator+(const Tensor& other) const {
  realize();
  other.realize();
  if (p->shape != other.p->shape)
    throw runtime_error("Tensor::operator+ shape mismatch");
  Tensor out(p->shape, 0.0f);
  for (int i=0; i<size(); i++)
    out.p->data[i] = p->data[i] + other.p->data[i];
  return out;
}

// multiply element-with-element
Tensor Tensor::operator*(const Tensor& other) const {
  if (p->shape!=other.p->shape)
    throw runtime_error("Tensor::operator* shape mismatch");
  realize();
  other.realize();
  Tensor out(p->shape);
  for (int i=0; i<size(); ++i) 
    out.p->data[i] = p->data[i] * other.p->data[i];
  return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
  if (p->shape != other.p->shape)
    throw runtime_error("Tensor::operator- shape mismatch");
  realize();
  other.realize();
  Tensor out(p->shape);
  for (int i=0; i<size(); ++i) 
    out.p->data[i] = p->data[i] - other.p->data[i];
  return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
  realize();
  other.realize();

  if (p->shape.size() != 2 || other.p->shape.size() != 2) 
    throw runtime_error("matmul requires 2D tensors");
  
  int A = rows(), B = cols();
  int C = other.cols();
  int B2 = other.rows();
  if (B != B2)
    throw runtime_error("matmul shape mismatch (A_cols != B_rows)");
  
  Tensor out(A, C);
  for (int row1=0; row1<A; ++row1) {
    for (int col2=0; col2<C; ++col2) {
      // row2 = col1
      for (int col1=0; col1<B; ++col1) {
        out.at(row1,col2) += at(row1,col1) * other.at(col1,col2);
      }
    }
  }
  return out;
}

Tensor Tensor::transpose() const {
  realize();

  if (p->shape.size() != 2)
    throw runtime_error("transpose requires 2D tensor");
  
  int R = rows(), C = cols();
  Tensor out(C, R);
  for (int r=0; r<R; ++r) {
    for (int c=0; c<C; ++c) {
      out.at(c,r) = at(r,c);
    }
  }
  return out;
}

// fill the Tensor with this value
void Tensor::fill(float v) {
  realize();
  std::fill(p->data.begin(), p->data.end(), v);
}

// Shape ops

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
  realize();
  int old_size = size();
  int neg_index = -1; // if >=0 then there is already a shape index of '-1'

  // new total number of elements
  int new_total = 1;
  for (int i=0; i<(int)new_shape.size(); i++) {
    if (new_shape[i] == -1) {
      // if there is already one -1 index => error
      if (neg_index==-1) 
        throw runtime_error("reshape: multiple -1 dims");
      neg_index = i;
    } else {
      new_total *= new_shape[i];
    }
  }

  // replace -1 in new_shape
  vector<int> final_shape = new_shape;
  if (neg_index != -1) {
    if (new_total == 0 || old_size % new_total != 0) 
      throw runtime_error("reshape: size mismatch for -1");
    final_shape[neg_index] = old_size / new_total;
  }

  int final_total = prod(final_shape);
  if (final_total != old_size)
    throw runtime_error("reshape: total size mismatch");
  
  Tensor out(final_shape);
  out.p->data = p->data;

  return out;
}

Tensor Tensor::flatten() const {
  return reshape({size()});
}

Tensor Tensor::sum(int axis) const {
  realize();
  if (ndim() == 1) {
    // sum of a 1D Tensor = scalar in 1D tensor
    float s = 0;
    for (float v: p->data) 
      s += v;
    return Tensor({s}); // Tensor of 1 element
  }

  if (ndim() != 2) // temporary for now
    throw runtime_error("sum(axis): only implemented for 1D or 2D tensors");
  
  int R = rows(), C = cols();
  // sum over rows => output shape (C,)
  if (axis == 0) {
    Tensor out(C);
    out.fill(0);
    for (int r=0; r<R; r++) {
      for (int c=0; c<C; c++) {
        out[c] += at(r,c);
      }
    }
    return out;
  }
  // sum over cols => output shape (R,)
  if (axis == 1) {
    Tensor out(R);
    out.fill(0);
    for(int r=0; r<R; r++) {
      for (int c=0; c<C; c++) {
        out[r] += at(r,c);
      }
    }
    return out;
  }

  throw runtime_error("sum(axis): axis must be 0 or 1 for 2D tensor");
}

// src: (C,); dst: (B,C) => pad: (1,C)
// src: (D,); dst: (A,B,D) => pad: (1,1,D)
// src: (X,1,1); dst: (X,Y,Z) => allowed if (1->Y and 1->Z)
Tensor Tensor::broadcast_to(const vector<int>& new_shape) const {
  realize();
  const auto& src_shape = p->shape;
  int src_ndim = src_shape.size(); // e.g. 1
  int dst_ndim = new_shape.size(); // e.g. 3

  // pad src shape on the left with 1s so ranks match
  // e.g. pad from (D,) to (1,1,D)
  vector<int> padded_src_shape(dst_ndim, 1); // (3,) of value 1 each
  for (int i=0; i<src_ndim; i++) {
    padded_src_shape[dst_ndim-src_ndim+i] = src_shape[i];
  }

  // validate broadcast rule
  for (int i=0; i<dst_ndim; i++) {
    if (padded_src_shape[i] != 1 && padded_src_shape[i] != new_shape[i]) {
      throw runtime_error("broadcast_to: incompatible dim");
    }
  }

  // allocate new shape
  Tensor out(new_shape);
  out.fill(0.0f);

  // number of elements in output tensor
  int total = out.size();
  int lead = dst_ndim - src_ndim; // number of (1,1,...) padding
  vector<int> idx(dst_ndim); // store location of that <flat> cell

  for (int flat = 0; flat < total; flat++) {
    // convert flat index to multi-index -> idx: [col,row,batch]
    int tmp = flat;
    for (int d=dst_ndim-1; d>=0; d--) {
      idx[d] = tmp % new_shape[d];
      tmp /= new_shape[d];
    }

    // map <flat> cell, to which location on source
    vector<int> src_idx(src_ndim);
    for (int d = 0; d<dst_ndim; d++) {
      int sub_shape = padded_src_shape[d];
      int mapped = (sub_shape==1) ? 0 : idx[d];
      // care only non-one 
      if (d >= dst_ndim - src_ndim) 
        src_idx[d-(dst_ndim-src_ndim)] = mapped;
    }

    int src_flat = 0;
    int stride = 1;
    // convert source locations to flat
    for (int d=src_ndim-1; d>=0; d--) {
      src_flat += src_idx[d] * stride;
      stride *= src_shape[d];
    }
    out.p->data[flat] = p->data[src_flat];
  }

  return out;
}

bool Tensor::requires_grad() const {
  return p->requires_grad;
}

void Tensor::set_requires_grad(bool v) {
  p->requires_grad = v;
}

Tensor Tensor::grad() const {
  Tensor g(p->shape);
  g.p->data = p->grad;
  return g;
}

void Tensor::zero_grad() {
  p->grad.assign(size(), 0.0f);
}

void Tensor::_set_grad_fn(const shared_ptr<Op>& op, const vector<Tensor>& parents) {
  p->grad_fn = op;
  p->parents = parents;
  p->is_leaf = false;
}

void Tensor::_backward_impl(const Tensor& grad_output) {
  realize();
  grad_output.realize();

  if (p->grad.empty())
    p->grad.assign(size(), 0.0f);
  for (int i=0; i<size(); i++) {
    // p->grad = {1,1,1} for (3,) vector
    p->grad[i] += grad_output.p->data[i];
  }
  // if Tensor is a leaf Tensor 
  if (!p->grad_fn)
    return;

  auto& inputs = p->parents;
  auto op = p->grad_fn;
  Tensor output(*this); // copy pointer for passing
  auto input_grads = op->backward(grad_output, inputs, output);
  if (input_grads.size() != inputs.size()) {
    throw runtime_error("backward: op returned wrong number of input grads");
  }
  // recurse
  for (size_t i=0; i<inputs.size(); ++i) {
    if (inputs[i].requires_grad()) {
      inputs[i]._backward_impl(input_grads[i]);
    }
  }
}

void Tensor::backward() {
  if (size()!=1)
    throw runtime_error("backward(): only on scalar");
  Tensor g({1.0f});
  _backward_impl(g);
}

void Tensor::backward(const Tensor& g) {
  _backward_impl(g);
}