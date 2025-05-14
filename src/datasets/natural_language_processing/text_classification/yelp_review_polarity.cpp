#include "datasets/natural_language_processing/text_classification/yelp_review_polarity.h"

namespace xt::data::datasets {

    YelpReviewPolarity::YelpReviewPolarity(const std::string &root): YelpReviewPolarity::YelpReviewPolarity(root, DataMode::TRAIN, false) {
    }

    YelpReviewPolarity::YelpReviewPolarity(const std::string &root, DataMode mode): YelpReviewPolarity::YelpReviewPolarity(root, mode, false) {
    }

    YelpReviewPolarity::YelpReviewPolarity(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("YelpReview: YelpReview not implemented");
    }


    YelpReviewPolarity::YelpReviewPolarity(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("YelpReview: YelpReview not implemented");
    }


}
