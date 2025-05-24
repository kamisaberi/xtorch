#include "include/datasets/computer_vision/stereo_matching/scene_flow_stereo.h"

namespace xt::data::datasets
{
    // ---------------------- SceneFlowStereo ---------------------- //

    SceneFlowStereo::SceneFlowStereo(const std::string& root): SceneFlowStereo::SceneFlowStereo(
        root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    SceneFlowStereo::SceneFlowStereo(const std::string& root, xt::datasets::DataMode mode): SceneFlowStereo::SceneFlowStereo(
        root, mode, false, nullptr, nullptr)
    {
    }

    SceneFlowStereo::SceneFlowStereo(const std::string& root, xt::datasets::DataMode mode, bool download) :
        SceneFlowStereo::SceneFlowStereo(
            root, mode, download, nullptr, nullptr)
    {
    }

    SceneFlowStereo::SceneFlowStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer) : SceneFlowStereo::SceneFlowStereo(
        root, mode, download, std::move(transformer), nullptr)
    {
    }

    SceneFlowStereo::SceneFlowStereo(const std::string& root, xt::datasets::DataMode mode, bool download,
                           std::unique_ptr<xt::Module> transformer, std::unique_ptr<xt::Module> target_transformer):
        xt::datasets::Dataset(mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources();
        load_data();

    }


    void SceneFlowStereo::load_data()
    {

    }

    void SceneFlowStereo::check_resources()
    {

    }
}
