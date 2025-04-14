#include "../../../include/datasets/image-classification/flowers.h"

namespace xt::data::datasets {

    Flowers102::Flowers102(const std::string &root): Flowers102::Flowers102(root, DataMode::TRAIN, false) {
    }

    Flowers102::Flowers102(const std::string &root, DataMode mode): Flowers102::Flowers102(root, mode, false) {
    }

    Flowers102::Flowers102(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Flowers102: Flowers102 not implemented");
    }


    Flowers102::Flowers102(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Flowers102: Flowers102 not implemented");
    }

}
