#pragma once
#include "datasets/base/base.h"


namespace xt::data::datasets {
    class YelpReviewPolarity : BaseDataset {
        public :
            explicit YelpReviewPolarity(const std::string &root);
        YelpReviewPolarity(const std::string &root, DataMode mode);
        YelpReviewPolarity(const std::string &root, DataMode mode , bool download);
        YelpReviewPolarity(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
