#include "tensor.h"
#include "ops.h"
#include <iostream>

int main() {
  // x and w are leaf tensors with requires_grad = true 
  Tensor x {1.0f, 2.0f, 3.0f};
  Tensor w {4.0f, 5.0f, 6.0f};
  x.set_requires_grad(true);
  w.set_requires_grad(true);

  return 0;
}