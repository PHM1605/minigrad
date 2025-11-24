#pragma once 
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <memory>

using namespace std;

class Op;

class Tensor {
public:
  // constructors
  Tensor(); // empty tensor
  explicit Tensor(int n); // 1D zeros
  Tensor(int rows, int cols);
  Tensor(const vector<int>& shape, float fill=0.0f); // create a Tensor of shape (m,n) with fill value = 0.0
  Tensor(const vector<int>& shape, const vector<float>& values); // create a Tensor of shape (m,n) with fill values = {2,3,5,1,0,4,...}
  explicit Tensor(initializer_list<float> list); // create a Tensor 1D from values = {2,3,5,1,0,4,...}
  Tensor(initializer_list<initializer_list<float>> mat); // create a Tensor 2D from mat = {{0,1},{2,4},{3,5}}

  // shape helpers
  int ndim() const;
  int size() const; // number of elements
  int rows() const; // only for 2D
  int cols() const; // only for 2D
  
  // access underlying buffers (read-only view)
  vector<int> shape() const;
  string shape_str() const;

  const vector<float>& data() const;
  // element access (2D)
  float& at(int r, int c); // for normal tensor
  float at(int r, int c) const; // for const tensor
  // element access (1D) OR flat-index-access (2D)
  float& operator[](int i); // for normal tensor
  float operator[](int i) const; // for const tensor

  void print() const;

  Tensor operator+(const Tensor& other) const; // element-wise (same shape)
  Tensor operator*(const Tensor& other) const; // element-wise (same shape)
  Tensor operator-(const Tensor& other) const; // element-wise (same shape)
  Tensor matmul(const Tensor& other) const;
  Tensor transpose() const;

  void fill(float v);

  Tensor reshape(const vector<int>& new_shape) const;
  Tensor flatten() const;
  Tensor sum(int axis) const;
  Tensor broadcast_to(const vector<int>& new_shape) const;

  // autograd API
  bool requires_grad() const;
  void set_requires_grad(bool v);

  // gradient
  Tensor grad() const;
  void zero_grad();

  void backward();
  void backward(const Tensor& grad_output); // backward with explicit grad_output

  // lazy 
  void realize() const;
  bool is_realized() const;

  // product of all elements in <v>
  static int prod(const vector<int>& v);

private:
  struct Impl {
    vector<int> shape; // for 1D: {n}; for 2D: {rows,cols}
    vector<float> data;
    vector<float> grad;
    bool requires_grad = false; // same size as <data>
    bool is_leaf = true;
    bool realized = false;

    shared_ptr<Op> grad_fn; // <op> that produce this tensor
    vector<Tensor> parents;
  };
  shared_ptr<Impl> p = make_shared<Impl>();

  void _set_grad_fn(const shared_ptr<Op>& op, const vector<Tensor>& parents);
  void _backward_impl(const Tensor& grad_output);

  // allow ops.cpp to touch internals
  friend Tensor add(const Tensor&, const Tensor&);
  friend Tensor sub(const Tensor&, const Tensor&);
  friend Tensor mul(const Tensor&, const Tensor&);
  friend Tensor matmul(const Tensor&, const Tensor&);
  friend Tensor reduce_sum(const Tensor&);
  friend Tensor broadcast(const Tensor&, const vector<int>&);
  friend Tensor relu(const Tensor&);
  friend Tensor sigmoid(const Tensor&);
  friend Tensor tanh_fn(const Tensor&);
  friend Tensor softmax(const Tensor&);
  friend Tensor cross_entropy(const Tensor&, const Tensor&);
};