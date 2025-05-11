#include "../../../include/datasets/natural_language_processing/text_classification/amazon_review_polarity.h"

namespace xt::data::datasets {

    AmazonReview::AmazonReview(const std::string &root): AmazonReview::AmazonReview(root, DataMode::TRAIN, false) {
    }

    AmazonReview::AmazonReview(const std::string &root, DataMode mode): AmazonReview::AmazonReview(root, mode, false) {
    }

    AmazonReview::AmazonReview(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("AmazonReview: AmazonReview not implemented");
    }


    AmazonReview::AmazonReview(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("AmazonReview: AmazonReview not implemented");
    }


}
