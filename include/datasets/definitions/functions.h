#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data {
    torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset , torch::data::transforms::Stack<>()>
    transform_dataset(xt::data::datasets::BaseDataset dataset, vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms);
}
