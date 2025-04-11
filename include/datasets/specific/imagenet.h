#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


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
