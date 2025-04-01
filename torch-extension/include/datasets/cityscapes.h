#pragma once
#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class Cityscapes : public BaseDataset {
    public :
        Cityscapes(const std::string &root);
        Cityscapes(const std::string &root, DataMode mode);
        Cityscapes(const std::string &root, DataMode mode , bool download);
        Cityscapes(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
