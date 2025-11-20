#include "tensor.h"
#include "ops.h"
#include <iostream>

int main(){
  Tensor x{1.0f, 2.0f, 3.0f};
  Tensor y{4.0f, 5.0f, 6.0f};

  Tensor z_add = add(x, y);
  cout << "add(x,y):\n";
  z_add.print();

  Tensor z_mul = mul(x, y);
  cout << "mul(x,y):\n";
  z_mul.print();

  Tensor A = {{1,2,3},
              {4,5,6}};
  Tensor B = {{7,8},
              {9,10},
              {11,12}};
  cout << "A @ B via matmul(op):\n";
  Tensor C = matmul(A, B);
  C.print();

  return 0;
}