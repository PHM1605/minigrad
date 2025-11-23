#pragma once 
#include "tensor.h"
#include "ops.h"

Tensor mse_loss(const Tensor& pred, const Tensor& target);

