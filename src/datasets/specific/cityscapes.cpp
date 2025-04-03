#include "../../../include/datasets/specific/cityscapes.h"

namespace xt::data::datasets {

    Cityscapes::Cityscapes(const std::string &root): Cityscapes::Cityscapes(root, DataMode::TRAIN, false) {
    }

    Cityscapes::Cityscapes(const std::string &root, DataMode mode): Cityscapes::Cityscapes(root, mode, false) {
    }

    Cityscapes::Cityscapes(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Cityscapes: Cityscapes not implemented");
    }


    Cityscapes::Cityscapes(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Cityscapes: Cityscapes not implemented");
    }

}
