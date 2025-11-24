#include "tensor.h"
#include "ops.h"
#include <iostream>

int main() {
  Tensor x {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f};
  x.set_requires_grad(true);
  Tensor y = relu(x);
  Tensor s = reduce_sum(y);
  s.backward();
  cout << "x:\n";
  x.print();
  cout << "relu(x):\n";
  y.print();
  cout << "grad x (dL/dx):\n";
  x.grad().print();

  return 0;
}