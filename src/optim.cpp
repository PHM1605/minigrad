#include "optim.h"

void SGD::step() {
  for (Tensor* p : params) {
    if (!p->requires_grad()) 
      continue;
    Tensor g = p->grad();
    // loop over all (flat) elements of grad Tensor
    for (int i=0; i<p->size(); i++)
      (*p)[i] -= lr * g[i];
  }
}

void SGD::zero_grad() {
  for (Tensor* p : params) {
    p->zero_grad();
  }
}