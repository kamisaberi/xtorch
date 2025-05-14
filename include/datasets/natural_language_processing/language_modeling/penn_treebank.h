#pragma once


#include "datasets/base/base.h"

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
