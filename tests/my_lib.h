#ifndef MY_LIB_H
#define MY_LIB_H

#include <torch/torch.h>

// A simple function that doubles the input tensor
torch::Tensor double_tensor(const torch::Tensor& input);

#endif // MY_LIB_H