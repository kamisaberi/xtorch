#include "datasets/natural_language_processing/text_classification/amazon_review_polarity.h"

namespace xt::data::datasets {

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string &root): AmazonReviewPolarity::AmazonReviewPolarity(root, DataMode::TRAIN, false) {
    }

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string &root, DataMode mode): AmazonReviewPolarity::AmazonReviewPolarity(root, mode, false) {
    }

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("AmazonReview: AmazonReview not implemented");
    }


    AmazonReviewPolarity::AmazonReviewPolarity(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("AmazonReview: AmazonReview not implemented");
    }


}
