#include "tensor.h"
#include <iostream>
#include <cassert>
#include <iomanip> // for format in Tensor::print()

// Constructors
Tensor::Tensor(): shape({0}), data() {}
Tensor::Tensor(int n): shape({n}), data(n, 0.0f) {}
Tensor::Tensor(int rows, int cols): shape({rows, cols}), data(rows*cols, 0.0f) {}

Tensor::Tensor(const vector<int>& shape_, float fill):
  shape(shape_) {
  int n = prod(shape); // number of elements in Tensor
  data.assign(n, fill);
}

Tensor::Tensor(const vector<int>& shape_, const vector<float>& values)
: shape(shape_) {
  int n = prod(shape);
  if ((int)values.size() != n) throw runtime_error("Tensor: values size != shape product");
  data = values;
}

Tensor::Tensor(initializer_list<float> list):
  shape({(int)list.size()}), data(list) {}

Tensor::Tensor(initializer_list<initializer_list<float>> mat) {
  int r = (int)mat.size();
  int c = r ? (int)mat.begin()->size() : 0;
  shape = {r, c};
  data.reserve(r*c);
  for (auto &row: mat) {
    // if any row length != 1st row length
    if ((int)row.size() != c) 
      throw runtime_error("Tensor: ragged initializer not allowed");
    for (float v: row)
      data.push_back(v);
  } 
}

int Tensor::ndim() const {
  return (int)shape.size();
}

// total number of elements in Tensor 
int Tensor::size() const {
  return prod(shape);
}

int Tensor::rows() const {
  if (shape.size() != 2)
    throw runtime_error("Tensor::rows() requires 2D tensor");
  return shape[0];
}

int Tensor::cols() const {
  if (shape.size() != 2)
    throw runtime_error("Tensor::cols() requires 2D tensor");
  return shape[1];
}

Tensor Tensor::reshape(const std::vector<int>& new_shape) const {
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
  out.data = data;

  return out;
}

Tensor Tensor::flatten() const {
  return reshape({size()});
}

Tensor Tensor::sum(int axis) const {
  if (ndim() == 1) {
    // sum of a 1D Tensor = scalar in 1D tensor
    float s = 0;
    for (float v: data) 
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

// shape: (3,) -> (2,3)
Tensor Tensor::broadcast_to(const vector<int>& new_shape) const {
  
}

string Tensor::shape_str() const {
  string s = "(";
  for (size_t i=0; i<shape.size(); ++i) {
    s += to_string(shape[i]);
    if (i+1 < shape.size())
      s += ",";
  }
  s += ")";
  return s;
}

// for non-const 2D version
float& Tensor::at(int r, int c) {
  int R = rows(), C = cols();
  if (r<0 || r>=R || c<0 || c>=C)
    throw out_of_range("Tensor::at index");
  return data[r*C+c];
} 
// for const 2D version
float Tensor::at(int r, int c) const {
  int R = rows(), C = cols();
  if (r<0 || r>=R || c<0 || c>=C)
    throw out_of_range("Tensor::at index");
  return data[r*C+c];
}

// for non-const 1D version
float& Tensor::operator[](int i) {
  if (i<0 || i>=(int)data.size())
    throw out_of_range("Tensor::operator[]");
  return data[i];
}
// for const 1D version
float Tensor::operator[](int i) const {
  if (i<0 || i>=(int)data.size())
    throw out_of_range("Tensor::operator[]");
  return data[i];
}

void Tensor::print() const {
  if (shape.size() == 1) {
    for (int i=0; i<size(); i++) {
      cout << data[i] << " ";
    }
    cout << "\n";
    return;
  }
  else if (shape.size() == 2) {
    int R = rows(), C = cols();
    for (int r=0; r < R; r++) {
      for (int c=0; c<C; c++) {
        cout << setw(6) << at(r,c) << " ";
      }
      cout << "\n";
    }
  } else {
    for (int i=0; i<size(); ++i) 
      cout << data[i] << " ";
    cout << "\n";
  }
}

Tensor Tensor::operator+(const Tensor& other) const {
  if (shape!=other.shape)
    throw runtime_error("Tensor::operator+ shape mismatch");
  Tensor out(shape, 0.0f);
  int n = size();
  for (int i=0; i<n; i++)
    out.data[i] = data[i] + other.data[i];
  return out;
}

// multiply element-with-element
Tensor Tensor::operator*(const Tensor& other) const {
  if (shape!=other.shape)
    throw runtime_error("Tensor::operator* shape mismatch");
  Tensor out(shape, 0.0);
  int n = size();
  for (int i=0; i<n; ++i) 
    out.data[i] = data[i] * other.data[i];
  return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
  if (shape.size() != 2 || other.shape.size() != 2) 
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
  if (shape.size() != 2)
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
  std::fill(data.begin(), data.end(), v);
}

// product of all elements in <v>
int Tensor::prod(const std::vector<int>& v) {
  if (v.empty())
    return 0;
  int p = 1;
  for (int x: v)
    p *= x;
  return p;
}
