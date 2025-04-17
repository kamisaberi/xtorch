#include "../../../include/datasets/image-classification/sun.h"

namespace xt::data::datasets {

    SUN397::SUN397(const std::string &root): SUN397::SUN397(root, DataMode::TRAIN, false) {
    }

    SUN397::SUN397(const std::string &root, DataMode mode): SUN397::SUN397(root, mode, false) {
    }

    SUN397::SUN397(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SUN397: SUN397 not implemented");
    }


    SUN397::SUN397(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SUN397: SUN397 not implemented");
    }


}
