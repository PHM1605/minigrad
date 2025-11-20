#include "tensor.h"
#include <iostream>

using namespace std;

int main() {
  Tensor A = {{1.0f, 2.0f, 3.0f},
              {4.0f, 5.0f, 6.0f}};
  Tensor B = {{7.0f, 8.0f},
              {9.0f, 10.0f},
              {11.0f, 12.0f}};
  
  cout << "A shape: " << A.shape_str() << "\n";
  cout << "B shape: " << B.shape_str() << "\n";

  Tensor C = A.matmul(B);
  cout << "A @ B =\n";
  C.print();

  cout << "\nTranspose of A:\n";
  Tensor At = A.transpose();
  At.print();

  return 0;
}