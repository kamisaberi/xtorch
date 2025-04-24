#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class VCTK : BaseDataset {
        public :
            explicit VCTK(const std::string &root);
        VCTK(const std::string &root, DataMode mode);
        VCTK(const std::string &root, DataMode mode , bool download);
        VCTK(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
