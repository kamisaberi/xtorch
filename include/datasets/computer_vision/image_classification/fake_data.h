#pragma once

#include "../../common.h"



using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class FakeData : public xt::datasets::Dataset
    {
    public :
        explicit FakeData(size_t size);
        FakeData(size_t size, xt::datasets::DataMode mode );
        FakeData(size_t size, xt::datasets::DataMode mode ,vector<int64_t> shape);
        FakeData(size_t size, xt::datasets::DataMode mode ,vector<int64_t> shape,
                   std::unique_ptr<xt::Module> transformer);
        FakeData(size_t size, xt::datasets::DataMode mode ,vector<int64_t> shape,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private :
        size_t size_;
        vector<int64_t> shape_ = {3, 24, 24};

        void generate_data();
    };
}
