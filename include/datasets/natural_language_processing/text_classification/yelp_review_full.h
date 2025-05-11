#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class YelpReview : BaseDataset {
        public :
            explicit YelpReview(const std::string &root);
        YelpReview(const std::string &root, DataMode mode);
        YelpReview(const std::string &root, DataMode mode , bool download);
        YelpReview(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
