#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class WNLI : BaseDataset {
        public :
            explicit WNLI(const std::string &root);
        WNLI(const std::string &root, DataMode mode);
        WNLI(const std::string &root, DataMode mode , bool download);
        WNLI(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
