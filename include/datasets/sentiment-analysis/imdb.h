#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class IMDB : BaseDataset {
        public :
            explicit IMDB(const std::string &root);
        IMDB(const std::string &root, DataMode mode);
        IMDB(const std::string &root, DataMode mode , bool download);
        IMDB(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
