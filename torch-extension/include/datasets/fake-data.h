#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class FakeData : public BaseDataset {
    public :
        FakeData(const std::string &root);

        FakeData(const std::string &root, DataMode mode);

        FakeData(const std::string &root, DataMode mode, size_t size);

        FakeData(const std::string &root, DataMode mode, size_t size, vector<int64_t> shape_);

        FakeData(const std::string &root, DataMode mode, size_t size, vector<int64_t> shape_,
                 vector<std::function<torch::Tensor(torch::Tensor)> > transforms);

    private :
        size_t size_;
        vector<int64_t> shape_ = {3, 24, 24};
        void generate_data();

    };
}
