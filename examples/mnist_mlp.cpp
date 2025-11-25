#include "dataset.h"
#include "module.h"
#include "layers.h"
#include "optim.h"
#include "ops.h"
#include "loss.h"
#include <iostream>

int main() {
  MNISTDataset train_ds("/media/phm/New Volume/Minh/data/mnist", true);
  MNISTDataset test_ds("/media/phm/New Volume/Minh/data/mnist", false);

  DataLoader train_loader(train_ds, 64, true);

  // MLP
  auto model = make_shared<Sequential>();
  model->add_module(make_shared<Dense>(784, 128));
  model->add_module(make_shared<ReLU>());
  model->add_module(make_shared<Dense>(128,128));
  model->add_module(make_shared<ReLU>());
  model->add_module(make_shared<Dense>(128, 10));

  SGD opt(model->parameters(), 0.01);

  for (int epoch=0; epoch<2; epoch++) {
    train_loader.reset();
    int batch = 0;
    Tensor X, Y;
    while(train_loader.next_batch(X, Y)) {
      Tensor logits = (*model)(X); // (B,10)
      Tensor loss = cross_entropy(logits, Y); // (B,)
      Tensor mean_loss = reduce_sum(loss);
      
      opt.zero_grad();
      mean_loss.backward();
      opt.step();
      if (batch % 100 == 0) {
        cout << "epoch " << epoch << " batch " << batch << " loss ";
        mean_loss.print();
      }
      batch++;
    }
  }

  // Test accuracy
  DataLoader test_loader(test_ds, 64, false);
  int correct = 0;
  int total = 0;
  Tensor X, Y;
  while (test_loader.next_batch(X, Y)) {
    Tensor logits = (*model)(X);
    Tensor prob = softmax(logits);
    // loop over each sample in batch
    for (int i=0; i<Y.size(); i++) {
      // argmax
      float best = -1e9;
      int best_c = -1;
      // loop over each class probs
      for (int c=0; c<10; c++) {
        float v = prob.at(i,c);
        if (v>best) {
          best = v;
          best_c = c;
        }
      }
      if (best_c == (int)Y[i])
        correct++;
      total++;
    }
  }
  cout << "Test accuracy: " << (100.0*correct/total) << "%\n";

  return 0;
}