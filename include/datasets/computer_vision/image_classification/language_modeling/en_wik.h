#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class EnWik : BaseDataset {
        public :
            explicit EnWik(const std::string &root);
        EnWik(const std::string &root, DataMode mode);
        EnWik(const std::string &root, DataMode mode , bool download);
        EnWik(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
