#pragma once 
#include <vector>
#include <memory>
#include <string>
#include "tensor.h"

class Module {
public:
  virtual ~Module() = default;

  virtual Tensor forward(const Tensor& x) = 0;
  Tensor operator()(const Tensor& x) {
    return forward(x);
  }
  virtual vector<Tensor*> parameters() {
    return {};
  }
  virtual void train() {}
  virtual void eval() {}
};

class Sequential: public Module {
public:
  Sequential() = default;
  void add_module(shared_ptr<Module> m) {
    modules.push_back(m);
  }
  Tensor forward(const Tensor& x) override;
  vector<Tensor*> parameters() override;
private:
  vector<shared_ptr<Module>> modules;
};

class ReLU: public Module {
public:
  Tensor forward(const Tensor& x) override;
};

class Sigmoid: public Module {
public:
  Tensor forward(const Tensor& x) override;
};

class TanhAct: public Module {
public:
  Tensor forward(const Tensor& x) override;
};