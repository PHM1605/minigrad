#include <iostream>
#include <vector>
#include "module.h"
#include "layers.h"
#include "optim.h"
#include "loss.h"
#include "ops.h"

int main() {
  auto seq = make_shared<Sequential>();
  seq->add_module(make_shared<Dense>(4, 8));
  seq->add_module(make_shared<ReLU>());
  seq->add_module(make_shared<Dense>(8, 8));
  seq->add_module(make_shared<ReLU>());
  seq->add_module(make_shared<Dense>(8, 1));

  vector<float> xs = {
    0.1f, 0.2f, 0.3f, 0.4f,
    0.5f, 0.6f, 0.7f, 0.8f,
    1.0f, 1.1f, 1.2f, 1.3f
  };
  Tensor X({3,4}, xs);
  vector<float> ys = {0.5f, -0.2f, 1.0f};
  Tensor Y({3,1}, ys);
  auto params = seq->parameters();
  SGD opt(params, 0.05f);

  for (int epoch=0; epoch<10; ++epoch) {
    Tensor pred = (*seq)(X);
    Tensor loss = mse_loss(pred, Y);
    opt.zero_grad();
    loss.backward();
    opt.step(); // update params with learning-rate

    cout << "epoch " << epoch << " loss: ";
    loss.print();
  }

  // print first layer weights and bias 
  auto p = seq->parameters();
  cout << "\nParameter count: " << p.size() << "\n";
  cout << "First param (W of first Dense):\n";
  p[0]->print();
  cout << "Grad of 1st param:\n";
  p[0]->grad().print();

  return 0;
}