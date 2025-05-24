#include "include/datasets/computer_vision/image_classification/caltech256.h"

namespace xt::datasets {


    Caltech256::Caltech256(const std::string& root): Caltech256::Caltech256(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Caltech256::Caltech256(const std::string& root, xt::datasets::DataMode mode): Caltech256::Caltech256(
        root, mode, false, nullptr, nullptr)
    {
    }

    Caltech256::Caltech256(const std::string& root, xt::datasets::DataMode mode, bool download) :
        Caltech256::Caltech256(
            root, mode, download, nullptr, nullptr)
    {
    }

    Caltech256::Caltech256(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : Caltech256::Caltech256(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    Caltech256::Caltech256(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void Caltech256::load_data()
    {

    }

    void Caltech256::check_resources()
    {

    }



}
