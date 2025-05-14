#pragma once

#include "datasets/base/base.h"

namespace xt::data::datasets {
    class UDPOS : BaseDataset {
        public :
            explicit UDPOS(const std::string &root);
        UDPOS(const std::string &root, DataMode mode);
        UDPOS(const std::string &root, DataMode mode , bool download);
        UDPOS(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
