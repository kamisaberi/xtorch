#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class IWSLT20169 : BaseDataset {
        public :
            explicit IWSLT20169(const std::string &root);
        IWSLT20169(const std::string &root, DataMode mode);
        IWSLT20169(const std::string &root, DataMode mode , bool download);
        IWSLT20169(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
