#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class YesNo : BaseDataset {
        public :
            explicit YesNo(const std::string &root);
        YesNo(const std::string &root, DataMode mode);
        YesNo(const std::string &root, DataMode mode , bool download);
        YesNo(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
