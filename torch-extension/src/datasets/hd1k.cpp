#include "../../include/datasets/hd1k.h"

namespace xt::data::datasets {

    HD1K::HD1K(const std::string &root): HD1K::HD1K(root, DataMode::TRAIN, false) {
    }

    HD1K::HD1K(const std::string &root, DataMode mode): HD1K::HD1K(root, mode, false) {
    }

    HD1K::HD1K(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }


    HD1K::HD1K(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }

}
