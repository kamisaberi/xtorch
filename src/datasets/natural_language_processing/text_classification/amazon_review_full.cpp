#include "datasets/natural_language_processing/text_classification/amazon_review_full.h"

namespace xt::data::datasets {

    AmazonReviewFull::AmazonReviewFull(const std::string &root): AmazonReviewFull::AmazonReviewFull(root, DataMode::TRAIN, false) {
    }

    AmazonReviewFull::AmazonReviewFull(const std::string &root, DataMode mode): AmazonReviewFull::AmazonReviewFull(root, mode, false) {
    }

    AmazonReviewFull::AmazonReviewFull(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("AmazonReview: AmazonReview not implemented");
    }


    AmazonReviewFull::AmazonReviewFull(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("AmazonReview: AmazonReview not implemented");
    }


}
