#include "datasets/computer_vision/stereo_matching/carla_stereo.h"

namespace xt::data::datasets {
    // ---------------------- CarlaStereo ---------------------- //
    CarlaStereo::CarlaStereo(const std::string &root): CarlaStereo::CarlaStereo(root, DataMode::TRAIN, false) {
    }

    CarlaStereo::CarlaStereo(const std::string &root, DataMode mode): CarlaStereo::CarlaStereo(root, mode, false) {
    }

    CarlaStereo::CarlaStereo(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CarlaStereo: CarlaStereo not implemented");
    }


    CarlaStereo::CarlaStereo(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CarlaStereo: CarlaStereo not implemented");
    }



}
