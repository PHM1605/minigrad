#include "loss.h"
#include <iostream>

Tensor mse_loss(const Tensor& pred, const Tensor& target) {
  Tensor diff = sub(pred, target);
  Tensor sq = mul(diff, diff);
  Tensor s = reduce_sum(sq);
  Tensor scale = Tensor({1.0f / pred.size()});
  Tensor loss = mul(s, scale);

  return loss;
}