#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class SogouNews : BaseDataset {
        public :
            explicit SogouNews(const std::string &root);
        SogouNews(const std::string &root, DataMode mode);
        SogouNews(const std::string &root, DataMode mode , bool download);
        SogouNews(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
