#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class MULTI30k : BaseDataset {
        public :
            explicit MULTI30k(const std::string &root);
        MULTI30k(const std::string &root, DataMode mode);
        MULTI30k(const std::string &root, DataMode mode , bool download);
        MULTI30k(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
