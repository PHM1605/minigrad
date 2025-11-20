#include "tensor.h"
#include <iostream>

int main() {
  Tensor A = {{1,2,3},
              {4,5,6}};
  cout << "A =\n";
  A.print();

  cout << "sum axis=0 (col sum):\n";
  A.sum(0).print();

  cout << "sum axis=1 (row sum):\n";
  A.sum(1).print();

  cout << "flatten:\n";
  A.flatten().print();

  cout << "reshape to (3,2):\n";
  A.reshape({3,2}).print();

  // Tensor b = {{10,20,30}}; // shape (3,)
  // Tensor bb = b.broadcast_to({2,3});
  // cout << "broadcast b to (2,3):\n";
  // bb.print();

  return 0;
}