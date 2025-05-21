#include "datasets/recommendation_systems/recommendation/movie_lens.h"

namespace xt::data::datasets
{
    // ---------------------- MovieLens ---------------------- //

    MovieLens::MovieLens(const std::string& root): MovieLens::MovieLens(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    MovieLens::MovieLens(const std::string& root, xt::datasets::DataMode mode): MovieLens::MovieLens(
        root, mode, false, nullptr, nullptr)
    {
    }

    MovieLens::MovieLens(const std::string& root, xt::datasets::DataMode mode, bool download) :
        MovieLens::MovieLens(
            root, mode, download, nullptr, nullptr)
    {
    }

    MovieLens::MovieLens(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : MovieLens::MovieLens(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    MovieLens::MovieLens(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void MovieLens::load_data()
    {

    }

    void MovieLens::check_resources()
    {

    }
}
