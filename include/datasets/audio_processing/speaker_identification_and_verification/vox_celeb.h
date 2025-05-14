#pragma once
#include "datasets/base/base.h"
#include "datasets/common.h"


namespace xt::data::datasets {
    class VoxCeleb : BaseDataset {
        public :
            explicit VoxCeleb(const std::string &root);
        VoxCeleb(const std::string &root, DataMode mode);
        VoxCeleb(const std::string &root, DataMode mode , bool download);
        VoxCeleb(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
