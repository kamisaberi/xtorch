#pragma once

#include "../../headers/datasets.h"
#include "../base.h"


namespace xt::data {
    torch::data::datasets::MapDataset<xt::data::datasets::BaseDataset , torch::data::transforms::Stack<>()>
    transform_dataset(xt::data::datasets::BaseDataset dataset, vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms);
}
