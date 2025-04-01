#pragma once
#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class DTD : public BaseDataset {
    public :
        DTD(const std::string &root);
        DTD(const std::string &root, DataMode mode);
        DTD(const std::string &root, DataMode mode , bool download);
        DTD(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
