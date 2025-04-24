#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class WikiText : BaseDataset {
        public :
            explicit WikiText(const std::string &root);
        WikiText(const std::string &root, DataMode mode);
        WikiText(const std::string &root, DataMode mode , bool download);
        WikiText(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
