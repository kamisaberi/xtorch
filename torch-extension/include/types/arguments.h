#pragma once
#include <iostream>
#include <type_traits>
#include <vector>
#include <torch/data.h>
#include <torch/csrc/autograd/grad_mode.h>
#include "enums.h"
using namespace std;


struct DatasetArguments {
    DataMode mode = DataMode::TRAIN;
    bool download = false;
    vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms  = {};

};
