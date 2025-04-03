#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class Cityscapes : public BaseDataset {
    public :
        Cityscapes(const std::string &root);
        Cityscapes(const std::string &root, DataMode mode);
        Cityscapes(const std::string &root, DataMode mode , bool download);
        Cityscapes(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private :
        void load_data();

        void check_resources();
    };
}
