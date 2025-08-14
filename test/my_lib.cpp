#include "my_lib.h"

torch::Tensor double_tensor(const torch::Tensor& input) {
    return input * 2;
}