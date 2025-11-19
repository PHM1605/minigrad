#pragma once 
#include <vector>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include "ops.h"

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
  void reshape(const vector<int>& new_shape);
  string shape_str() const;

  // element access
  

  Tensor operator+(const Tensor& other) const;
  Tensor operator*(const Tensor& other) const;
  Tensor logical_xor(const Tensor& other) const;

  // Graph fields for autograd. E.g. c = add(a,b) => we consider <c> here
  Op* op = nullptr; // <add>
  vector<Tensor*> parents; // pointers to <a>,<b> (c's parents)
  bool requires_grad = false;
  Tensor* grad = nullptr; // gradient tensor of <c>

  void print() const;
};