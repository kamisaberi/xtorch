#pragma once
#include "datasets/base/base.h"

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
