#include "tensor.h"
#include "ops.h"
#include "layers.h"
#include "optim.h"
#include "loss.h"
#include <iostream>

int main() {
  // simple dataset y = 2x+1
  vector<float> xs, ys;
  for (int i=0; i<100; i++) {
    float x = i/100.0f;
    float y = 2*x+1;
    xs.push_back(x);
    ys.push_back(y);
  }
  Tensor X({100,1}, xs);
  Tensor Y({100,1}, ys);

  X.set_requires_grad(false);
  Y.set_requires_grad(false);

  Linear model(1,1);
  SGD opt({&model.W, &model.b}, 0.1f);

  for (int epoch = 0; epoch<200; epoch++) {
    Tensor pred = model(X);
    Tensor loss = mse_loss(pred, Y);
    opt.zero_grad();
    loss.backward();
    auto gW = model.W.grad();
    auto gB = model.b.grad();
    float sumW = 0, sumB = 0;
    for (float v: gW.data())
      sumW += v;
    for (float v: gB.data())
      sumB += v;
    std::cerr << "[grad] W.sum=" << sumW << " b.sum=" << sumB << endl;

    opt.step();
    if (epoch % 20 == 0) {
      cout << "epoch " << epoch << " loss = ";
      loss.print();
    }
  }

  cout << "Learned W, b:\n";
  model.W.print();
  model.b.print();

  return 0;
}