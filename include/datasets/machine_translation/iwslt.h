#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class IWSLT : BaseDataset {
        public :
            explicit IWSLT(const std::string &root);
        IWSLT(const std::string &root, DataMode mode);
        IWSLT(const std::string &root, DataMode mode , bool download);
        IWSLT(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
