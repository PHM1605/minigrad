#include "module.h"
#include "layers.h"
#include "loss.h"
#include "ops.h"
#include "optim.h"
#include <iostream>

// Toy dataset: 3 classes, 2 features
// class 0 near (0,0)
// class 1 near (1,0)
// class 2 near (0,1)
int main() {
  vector<float> Xv = {
    0.0, 0.1, // class 0
    0.1, -0.1, // class 0
    1.0, 0.1, // class 1
    0.9, -0.1, // class 1
    0.0, 1.0, // class 2
    -0.1, 1.1 // class 2
  };
  Tensor X({6,2}, Xv);
  Tensor Y({0, 0, 1, 1, 2, 2});

  // MLP
  auto model = make_shared<Sequential>();
  model->add_module(make_shared<Dense>(2,16));
  model->add_module(make_shared<ReLU>());
  model->add_module(make_shared<Dense>(16,3));

  SGD opt(model->parameters(), 0.1f);

  for (int epoch=0; epoch<100; epoch++) {
    Tensor logits = (*model)(X); // (6,3)
    Tensor loss = cross_entropy(logits, Y); // (6,)
    Tensor mean_loss = reduce_sum(loss); // (1,)
    opt.zero_grad();
    mean_loss.backward();
    opt.step();
    
    if (epoch % 10 == 0) {
      float L = loss.data()[0];
      cout << "epoch: " << epoch << "\nloss\n";
      mean_loss.print();
    } 
  }
  // final prediction probs
  Tensor pred = softmax((*model)(X));
  cout << "\nPredicted class probs:\n";
  pred.print();

  return 0;
}