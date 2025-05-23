#include "datasets/natural_language_processing/text_classification/amazon_review_polarity.h"


namespace xt::data::datasets
{
    // ---------------------- AmazonReviewPolarity ---------------------- //

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string& root): AmazonReviewPolarity::AmazonReviewPolarity(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string& root, xt::datasets::DataMode mode): AmazonReviewPolarity::AmazonReviewPolarity(
        root, mode, false, nullptr, nullptr)
    {
    }

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string& root, xt::datasets::DataMode mode, bool download) :
        AmazonReviewPolarity::AmazonReviewPolarity(
            root, mode, download, nullptr, nullptr)
    {
    }

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : AmazonReviewPolarity::AmazonReviewPolarity(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    AmazonReviewPolarity::AmazonReviewPolarity(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void AmazonReviewPolarity::load_data()
    {

    }

    void AmazonReviewPolarity::check_resources()
    {

    }
}
