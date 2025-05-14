#include "../../include/models/base.h"

namespace xt::models {

    BaseModel::BaseModel() = default;
    Model::Model(int a):BaseModel()  {
      this->a = a;
    }

}