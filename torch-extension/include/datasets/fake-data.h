#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class FakeData : public BaseDataset {
    public :
        FakeData(const std::string &root);

        FakeData(const std::string &root, DataMode mode);

        FakeData(const std::string &root, DataMode mode, bool download);

        FakeData(const std::string &root, DataMode mode, bool download,
                 vector<std::function<torch::Tensor(torch::Tensor)> > transforms);

    private :
        size_t size_;
        vector<int64_t> shape_ = {3, 24, 24};

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
