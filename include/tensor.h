#pragma once 
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>

using namespace std;

class Tensor {
public:
  // for 1D: {n}; for 2D: {rows,cols}
  vector<int> shape;
  vector<float> data;

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
  int size() const;
  int rows() const; // only for 2D
  int cols() const; // only for 2D
  Tensor reshape(const vector<int>& new_shape) const;
  Tensor flatten() const;
  Tensor sum(int axis) const;
  Tensor broadcast_to(const vector<int>& new_shape) const;

  string shape_str() const;

  // element access (2D)
  float& at(int r, int c); // for normal tensor
  float at(int r, int c) const; // for const tensor
  // element access (1D)
  float& operator[](int i); // for normal tensor
  float operator[](int i) const; // for const tensor

  void print() const;

  Tensor operator+(const Tensor& other) const; // element-wise (same shape)
  Tensor operator*(const Tensor& other) const; // element-wise (same shape)

  Tensor matmul(const Tensor& other) const;
  Tensor transpose() const;

  void fill(float v);


  // product of all elements in <v>
  static int prod(const vector<int>& v);
};