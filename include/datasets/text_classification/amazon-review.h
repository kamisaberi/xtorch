#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class AmazonReview : BaseDataset {
        public :
            explicit AmazonReview(const std::string &root);
        AmazonReview(const std::string &root, DataMode mode);
        AmazonReview(const std::string &root, DataMode mode , bool download);
        AmazonReview(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
