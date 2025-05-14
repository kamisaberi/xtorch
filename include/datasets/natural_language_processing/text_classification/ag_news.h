#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets {
    class AgNews : BaseDataset {
        public :
            explicit AgNews(const std::string &root);
        AgNews(const std::string &root, DataMode mode);
        AgNews(const std::string &root, DataMode mode , bool download);
        AgNews(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
