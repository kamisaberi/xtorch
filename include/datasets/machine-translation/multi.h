#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class MULTI : BaseDataset {
        public :
            explicit MULTI(const std::string &root);
        MULTI(const std::string &root, DataMode mode);
        MULTI(const std::string &root, DataMode mode , bool download);
        MULTI(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
