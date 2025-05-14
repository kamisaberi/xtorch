#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class IWSLT2017 : BaseDataset {
        public :
            explicit IWSLT2017(const std::string &root);
        IWSLT2017(const std::string &root, DataMode mode);
        IWSLT2017(const std::string &root, DataMode mode , bool download);
        IWSLT2017(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
