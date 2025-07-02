#include "include/datasets/natural_language_processing/text_classification/amazon_review_full.h"

namespace xt::datasets
{
    // ---------------------- AmazonReviewFull ---------------------- //

    AmazonReviewFull::AmazonReviewFull(const std::string& root): AmazonReviewFull::AmazonReviewFull(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    AmazonReviewFull::AmazonReviewFull(const std::string& root, xt::datasets::DataMode mode): AmazonReviewFull::AmazonReviewFull(
        root, mode, false, nullptr, nullptr)
    {
    }

    AmazonReviewFull::AmazonReviewFull(const std::string& root, xt::datasets::DataMode mode, bool download) :
        AmazonReviewFull::AmazonReviewFull(
            root, mode, download, nullptr, nullptr)
    {
    }

    AmazonReviewFull::AmazonReviewFull(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : AmazonReviewFull::AmazonReviewFull(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    AmazonReviewFull::AmazonReviewFull(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void AmazonReviewFull::load_data()
    {

    }

    void AmazonReviewFull::check_resources()
    {

    }
}
