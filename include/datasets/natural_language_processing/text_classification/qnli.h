#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class QNLI : BaseDataset {
        public :
            explicit QNLI(const std::string &root);
        QNLI(const std::string &root, DataMode mode);
        QNLI(const std::string &root, DataMode mode , bool download);
        QNLI(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
