#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


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
