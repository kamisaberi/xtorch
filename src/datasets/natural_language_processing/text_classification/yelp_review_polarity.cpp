#include "include/datasets/natural_language_processing/text_classification/yelp_review_polarity.h"


namespace xt::datasets
{
    // ---------------------- YelpReviewPolarity ---------------------- //

    YelpReviewPolarity::YelpReviewPolarity(const std::string& root): YelpReviewPolarity::YelpReviewPolarity(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    YelpReviewPolarity::YelpReviewPolarity(const std::string& root, xt::datasets::DataMode mode): YelpReviewPolarity::YelpReviewPolarity(
        root, mode, false, nullptr, nullptr)
    {
    }

    YelpReviewPolarity::YelpReviewPolarity(const std::string& root, xt::datasets::DataMode mode, bool download) :
        YelpReviewPolarity::YelpReviewPolarity(
            root, mode, download, nullptr, nullptr)
    {
    }

    YelpReviewPolarity::YelpReviewPolarity(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : YelpReviewPolarity::YelpReviewPolarity(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    YelpReviewPolarity::YelpReviewPolarity(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void YelpReviewPolarity::load_data()
    {

    }

    void YelpReviewPolarity::check_resources()
    {

    }
}
