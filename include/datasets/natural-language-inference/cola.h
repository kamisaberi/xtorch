#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class COLA : BaseDataset {
        public :
            explicit COLA(const std::string &root);
        COLA(const std::string &root, DataMode mode);
        COLA(const std::string &root, DataMode mode , bool download);
        COLA(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
