#include "datasets/computer_vision/image_classification/sbu.h"

namespace xt::data::datasets
{
    // ---------------------- SBU ---------------------- //

    SBU::SBU(const std::string& root): SBU::SBU(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SBU::SBU(const std::string& root, xt::datasets::DataMode mode): SBU::SBU(
        root, mode, false, nullptr, nullptr)
    {
    }

    SBU::SBU(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SBU::SBU(
            root, mode, download, nullptr, nullptr)
    {
    }

    SBU::SBU(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SBU::SBU(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SBU::SBU(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SBU::load_data()
    {

    }

    void SBU::check_resources()
    {

    }
}
