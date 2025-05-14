#pragma once
#include "../../base/base.h"
#include "../../../headers/datasets.h"


namespace xt::data::datasets {
    class AmazonReviewPolarity : BaseDataset {
        public :
            explicit AmazonReviewPolarity(const std::string &root);
        AmazonReviewPolarity(const std::string &root, DataMode mode);
        AmazonReviewPolarity(const std::string &root, DataMode mode , bool download);
        AmazonReviewPolarity(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
