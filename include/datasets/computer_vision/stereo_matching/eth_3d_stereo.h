#pragma once

#include "datasets/base/base.h"

namespace xt::data::datasets {
    class ETH3DStereo : public BaseDataset {
    public:
        explicit ETH3DStereo(const std::string &root);
        ETH3DStereo(const std::string &root, DataMode mode);
        ETH3DStereo(const std::string &root, DataMode mode , bool download);
        ETH3DStereo(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
