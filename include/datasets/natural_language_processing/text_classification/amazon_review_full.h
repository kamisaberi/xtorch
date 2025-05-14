#pragma once
#include "datasets/base/base.h"

namespace xt::data::datasets
{
    class AmazonReviewFull : BaseDataset
    {
    public :
        explicit AmazonReviewFull(const std::string& root);
        AmazonReviewFull(const std::string& root, DataMode mode);
        AmazonReviewFull(const std::string& root, DataMode mode, bool download);
        AmazonReviewFull(const std::string& root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
