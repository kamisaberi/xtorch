#include "datasets/computer_vision/stereo_matching/eth_3d_stereo.h"

namespace xt::data::datasets
{
    // ---------------------- ETH3DStereo ---------------------- //

    ETH3DStereo::ETH3DStereo(const std::string& root): ETH3DStereo::ETH3DStereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ETH3DStereo::ETH3DStereo(const std::string& root, xt::datasets::DataMode mode): ETH3DStereo::ETH3DStereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    ETH3DStereo::ETH3DStereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ETH3DStereo::ETH3DStereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    ETH3DStereo::ETH3DStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ETH3DStereo::ETH3DStereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ETH3DStereo::ETH3DStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ETH3DStereo::load_data()
    {

    }

    void ETH3DStereo::check_resources()
    {

    }
}
