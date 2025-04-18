#pragma once

#include <torch/torch.h>

namespace xt::temp {

class TestDataset {

  public:
    TestDataset();
    ~TestDataset();
    torch::Tensor forward(torch::Tensor input);
};

}

