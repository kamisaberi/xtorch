#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


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
