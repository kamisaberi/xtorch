#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class PennTreebank : BaseDataset {
        public :
            explicit PennTreebank(const std::string &root);
        PennTreebank(const std::string &root, DataMode mode);
        PennTreebank(const std::string &root, DataMode mode , bool download);
        PennTreebank(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
