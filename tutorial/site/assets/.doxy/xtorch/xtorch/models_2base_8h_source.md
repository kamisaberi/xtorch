

# File base.h

[**File List**](files.md) **>** [**include**](dir_d44c64559bbebec7f509842c48db8b23.md) **>** [**models**](dir_828b612f8450ccb3091aade92090c8e3.md) **>** [**base.h**](models_2base_8h.md)

[Go to the documentation of this file](models_2base_8h.md)


```C++
#pragma once
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace xt::models {

class BaseModel: public torch::nn::Module {
  public:
  BaseModel();
   virtual  torch::Tensor forward(torch::Tensor input) const = 0;
};


class Model : public BaseModel {


  public:
    Model(int a);
    torch::Tensor forward(torch::Tensor input) const override;

private :
    int a;
};


}
```


