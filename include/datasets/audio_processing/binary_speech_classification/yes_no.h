#pragma once
#include "datasets/base/base.h"
// #include "datasets/common.h"


namespace xt::data::datasets
{
    class YesNo : public BaseDataset
    {
    public :
        explicit YesNo(const std::string& root);
        YesNo(const std::string& root, DataMode mode);
        YesNo(const std::string& root, DataMode mode, bool download);
        YesNo(const std::string& root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
