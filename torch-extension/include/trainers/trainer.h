#pragma once

#include <torch/torch.h>
#include <iostream>
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;


namespace xt {

template <typename Model , typename Loader>
class Trainer {

  public:
    explicit Trainer();
    void fit(Model &model , Loader &train_loader , Loader &test_loader);
  private:


}




}


