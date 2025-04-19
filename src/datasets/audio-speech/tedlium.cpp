#include "../../../include/datasets/audio-speech/tedlium.h"

namespace xt::data::datasets {

    Tedlium::Tedlium(const std::string &root): Tedlium::Tedlium(root, DataMode::TRAIN, false) {
    }

    Tedlium::Tedlium(const std::string &root, DataMode mode): Tedlium::Tedlium(root, mode, false) {
    }

    Tedlium::Tedlium(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Tedlium: Tedlium not implemented");
    }


    Tedlium::Tedlium(const std::string &root, DataMode mode, bool download,
                         TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Tedlium: Tedlium not implemented");
    }


}
