#pragma once
#include "datasets/base/base.h"

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
