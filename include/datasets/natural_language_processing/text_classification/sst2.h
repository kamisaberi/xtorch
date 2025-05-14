#pragma once
#include "../../base/base.h"
#include "../../../headers/datasets.h"


namespace xt::data::datasets {
    class SST2 : BaseDataset {
        public :
            explicit SST2(const std::string &root);
        SST2(const std::string &root, DataMode mode);
        SST2(const std::string &root, DataMode mode , bool download);
        SST2(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
