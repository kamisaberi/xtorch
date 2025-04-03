#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class FakeData : public BaseDataset {
    public :
        FakeData();

        FakeData(size_t size);

        FakeData( size_t size, vector<int64_t> shape);

        FakeData( size_t size, vector<int64_t> shape, TransformType transforms);

    private :
        size_t size_;
        vector<int64_t> shape_ = {3, 24, 24};

        void generate_data();
    };
}
