#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class ImageNet : BaseDataset {
    public :
        explicit  ImageNet(const std::string &root);
        ImageNet(const std::string &root, DataMode mode);
        ImageNet(const std::string &root, DataMode mode , bool download);
        ImageNet(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };
}
