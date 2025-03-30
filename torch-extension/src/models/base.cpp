#include "../../include/models/base.h"

namespace torch::ext::models {

    BaseModel::BaseModel() = default;
    Model::Model():BaseModel() {}

    torch::Tensor Model::forward(torch::Tensor input) const {
      std::cout << input << std::endl;
      return input * 3;
    }
}