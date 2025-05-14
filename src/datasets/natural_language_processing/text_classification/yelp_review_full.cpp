#include "datasets/natural_language_processing/text_classification/yelp_review_full.h"

namespace xt::data::datasets {

    YelpReviewFull::YelpReviewFull(const std::string &root): YelpReviewFull::YelpReviewFull(root, DataMode::TRAIN, false) {
    }

    YelpReviewFull::YelpReviewFull(const std::string &root, DataMode mode): YelpReviewFull::YelpReviewFull(root, mode, false) {
    }

    YelpReviewFull::YelpReviewFull(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("YelpReview: YelpReview not implemented");
    }


    YelpReviewFull::YelpReviewFull(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("YelpReview: YelpReview not implemented");
    }


}
