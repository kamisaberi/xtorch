#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class SNLI : BaseDataset {
        public :
            explicit SNLI(const std::string &root);
        SNLI(const std::string &root, DataMode mode);
        SNLI(const std::string &root, DataMode mode , bool download);
        SNLI(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
