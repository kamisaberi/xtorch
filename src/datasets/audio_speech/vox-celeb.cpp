#include "../../../include/datasets/audio-speech/vox-celeb.h"

namespace xt::data::datasets {

    VoxCeleb::VoxCeleb(const std::string &root): VoxCeleb::VoxCeleb(root, DataMode::TRAIN, false) {
    }

    VoxCeleb::VoxCeleb(const std::string &root, DataMode mode): VoxCeleb::VoxCeleb(root, mode, false) {
    }

    VoxCeleb::VoxCeleb(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("VoxCeleb: VoxCeleb not implemented");
    }


    VoxCeleb::VoxCeleb(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("VoxCeleb: VoxCeleb not implemented");
    }


}
