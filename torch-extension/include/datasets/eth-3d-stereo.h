#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class ETH3DStereo : public BaseDataset {
    public:
        ETH3DStereo(const std::string &root);
        ETH3DStereo(const std::string &root, DataMode mode);
        ETH3DStereo(const std::string &root, DataMode mode , bool download);
        ETH3DStereo(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        void load_data();

        void check_resources();
    };
}
