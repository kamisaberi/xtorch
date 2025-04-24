#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class CoNLL2000Chunking : BaseDataset {
        public :
            explicit CoNLL2000Chunking(const std::string &root);
        CoNLL2000Chunking(const std::string &root, DataMode mode);
        CoNLL2000Chunking(const std::string &root, DataMode mode , bool download);
        CoNLL2000Chunking(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
