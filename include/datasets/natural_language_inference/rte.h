#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class RTE : BaseDataset {
        public :
            explicit RTE(const std::string &root);
        RTE(const std::string &root, DataMode mode);
        RTE(const std::string &root, DataMode mode , bool download);
        RTE(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
