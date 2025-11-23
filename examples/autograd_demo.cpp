#include "tensor.h"
#include "ops.h"
#include <iostream>

int main() {
  // x and w are leaf tensors with requires_grad = true 
  Tensor x {1.0f, 2.0f, 3.0f};
  Tensor w {4.0f, 5.0f, 6.0f};
  x.set_requires_grad(true);
  w.set_requires_grad(true);
  // z = x*w (element-wise)
  Tensor z = mul(x, w);
  // loss = sum(z)
  Tensor loss = reduce_sum(z);

  cout << "x: ";
  x.print();
  cout << "w: ";
  w.print();
  cout << "z = x * w: ";
  z.print();
  cout << "loss = sum(z): ";
  loss.print();

  // backprop
  loss.backward();

  cout << "\nGrad x (dL/dx), expected [4,5,6]: ";
  x.grad().print();
  cout << "Grad w (dL/dw), expected [1,2,3]: ";
  w.grad().print();

  return 0;
}