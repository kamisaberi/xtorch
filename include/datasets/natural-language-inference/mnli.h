#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class MNLI : BaseDataset {
        public :
            explicit MNLI(const std::string &root);
        MNLI(const std::string &root, DataMode mode);
        MNLI(const std::string &root, DataMode mode , bool download);
        MNLI(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
