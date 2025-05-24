#include "include/datasets/computer_vision/semantic_segmentation/scannet.h"

namespace xt::data::datasets
{
    // ---------------------- ScanNet ---------------------- //

    ScanNet::ScanNet(const std::string& root): ScanNet::ScanNet(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    ScanNet::ScanNet(const std::string& root, xt::datasets::DataMode mode): ScanNet::ScanNet(
        root, mode, false, nullptr, nullptr)
    {
    }

    ScanNet::ScanNet(const std::string& root, xt::datasets::DataMode mode, bool download) :
        ScanNet::ScanNet(
            root, mode, download, nullptr, nullptr)
    {
    }

    ScanNet::ScanNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : ScanNet::ScanNet(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    ScanNet::ScanNet(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void ScanNet::load_data()
    {

    }

    void ScanNet::check_resources()
    {

    }
}
