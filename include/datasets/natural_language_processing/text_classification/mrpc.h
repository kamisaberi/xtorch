#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class MRPC : BaseDataset {
        public :
            explicit MRPC(const std::string &root);
        MRPC(const std::string &root, DataMode mode);
        MRPC(const std::string &root, DataMode mode , bool download);
        MRPC(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
