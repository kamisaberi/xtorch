#include "../../../include/datasets/text-classification/yelp-review.h"

namespace xt::data::datasets {

    YelpReview::YelpReview(const std::string &root): YelpReview::YelpReview(root, DataMode::TRAIN, false) {
    }

    YelpReview::YelpReview(const std::string &root, DataMode mode): YelpReview::YelpReview(root, mode, false) {
    }

    YelpReview::YelpReview(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("YelpReview: YelpReview not implemented");
    }


    YelpReview::YelpReview(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("YelpReview: YelpReview not implemented");
    }


}
