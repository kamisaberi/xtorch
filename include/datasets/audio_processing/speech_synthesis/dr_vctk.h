#pragma once
#include "datasets/base/base.h"
#include "datasets/common.h"


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
