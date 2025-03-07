#pragma once

#include "../base/datasets.h"


namespace torch::ext::data {
    torch::data::datasets::MapDataset<torch::ext::data::datasets::BaseDataset , torch::data::transforms::Stack<>()>
    transform_dataset(torch::ext::data::datasets::BaseDataset dataset, vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms);
}
