#pragma once
#include "datasets/base/base.h"


namespace xt::data::datasets
{
    class YelpReviewFull : BaseDataset
    {
    public :
        explicit YelpReviewFull(const std::string& root);
        YelpReviewFull(const std::string& root, DataMode mode);
        YelpReviewFull(const std::string& root, DataMode mode, bool download);
        YelpReviewFull(const std::string& root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
