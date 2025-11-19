#include "tensor.h"
#include <iostream>

using namespace std;

int main() {
  Tensor a({1.0f, 0.0f, 1.0f, 0.0f});
  Tensor b({0.0f, 1.0f, 1.0f, 0.0f});
  Tensor c = a * b;
  Tensor d = a + b;
  Tensor e = a.logical_xor(b);

  cout << "a*b:\n";
  c.print();
  cout << "a+b:\n";
  d.print();
  cout << "a XOR b:\n";
  e.print();

  return 0;
}