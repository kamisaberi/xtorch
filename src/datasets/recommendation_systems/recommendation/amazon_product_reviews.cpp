#include "include/datasets/recommendation_systems/recommendation/amazon_product_reviews.h"

namespace xt::datasets
{
    // ---------------------- AmazonProductReviews ---------------------- //

    AmazonProductReviews::AmazonProductReviews(const std::string& root): AmazonProductReviews::AmazonProductReviews(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    AmazonProductReviews::AmazonProductReviews(const std::string& root, xt::datasets::DataMode mode): AmazonProductReviews::AmazonProductReviews(
        root, mode, false, nullptr, nullptr)
    {
    }

    AmazonProductReviews::AmazonProductReviews(const std::string& root, xt::datasets::DataMode mode, bool download) :
        AmazonProductReviews::AmazonProductReviews(
            root, mode, download, nullptr, nullptr)
    {
    }

    AmazonProductReviews::AmazonProductReviews(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : AmazonProductReviews::AmazonProductReviews(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    AmazonProductReviews::AmazonProductReviews(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void AmazonProductReviews::load_data()
    {

    }

    void AmazonProductReviews::check_resources()
    {

    }
}
