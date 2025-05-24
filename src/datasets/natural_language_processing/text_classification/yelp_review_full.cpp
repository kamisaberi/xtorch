#include "include/datasets/natural_language_processing/text_classification/yelp_review_full.h"


namespace xt::data::datasets
{
    // ---------------------- YelpReviewFull ---------------------- //

    YelpReviewFull::YelpReviewFull(const std::string& root): YelpReviewFull::YelpReviewFull(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    YelpReviewFull::YelpReviewFull(const std::string& root, xt::datasets::DataMode mode): YelpReviewFull::YelpReviewFull(
        root, mode, false, nullptr, nullptr)
    {
    }

    YelpReviewFull::YelpReviewFull(const std::string& root, xt::datasets::DataMode mode, bool download) :
        YelpReviewFull::YelpReviewFull(
            root, mode, download, nullptr, nullptr)
    {
    }

    YelpReviewFull::YelpReviewFull(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : YelpReviewFull::YelpReviewFull(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    YelpReviewFull::YelpReviewFull(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void YelpReviewFull::load_data()
    {

    }

    void YelpReviewFull::check_resources()
    {

    }
}
