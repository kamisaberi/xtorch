#pragma once
#include <torch/torch.h>
#include <iostream>
#include <torch/script.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace torch::ext::models {

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