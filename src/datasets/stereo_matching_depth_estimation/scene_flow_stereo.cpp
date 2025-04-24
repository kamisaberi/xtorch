#include "../../../include/datasets/stereo_matching_depth_estimation/scene_flow_stereo.h"

namespace xt::data::datasets {

    SceneFlowStereo::SceneFlowStereo(const std::string &root): SceneFlowStereo::SceneFlowStereo(root, DataMode::TRAIN, false) {
    }

    SceneFlowStereo::SceneFlowStereo(const std::string &root, DataMode mode): SceneFlowStereo::SceneFlowStereo(root, mode, false) {
    }

    SceneFlowStereo::SceneFlowStereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SceneFlowStereo: SceneFlowStereo not implemented");
    }


    SceneFlowStereo::SceneFlowStereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SceneFlowStereo: SceneFlowStereo not implemented");
    }


}
