#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class STSB : BaseDataset {
        public :
            explicit STSB(const std::string &root);
        STSB(const std::string &root, DataMode mode);
        STSB(const std::string &root, DataMode mode , bool download);
        STSB(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
