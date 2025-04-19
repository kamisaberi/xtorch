#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class SST : BaseDataset {
        public :
            explicit SST(const std::string &root);
        SST(const std::string &root, DataMode mode);
        SST(const std::string &root, DataMode mode , bool download);
        SST(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
