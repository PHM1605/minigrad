#include "module.h"
#include "ops.h"

Tensor Sequential::forward(const Tensor& x) {
  Tensor out = x;
  for (auto&m: modules)
    out = m->forward(out);
  return out;
}

vector<Tensor*> Sequential::parameters() {
  vector<Tensor*> params;
  for (auto &m: modules) {
    auto p = m->parameters();
    params.insert(params.end(), p.begin(), p.end());
  }
  return params;
}

Tensor ReLU::forward(const Tensor& x) {
  return relu(x);
}

Tensor Sigmoid::forward(const Tensor& x) {
  return sigmoid(x);
}

Tensor TanhAct::forward(const Tensor& x) {
  return tanh_fn(x);
}