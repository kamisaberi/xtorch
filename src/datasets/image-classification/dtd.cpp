#include "../../../include/datasets/image-classification/dtd.h"

namespace xt::data::datasets {

    DTD::DTD(const std::string &root): DTD::DTD(root, DataMode::TRAIN, false) {
    }

    DTD::DTD(const std::string &root, DataMode mode): DTD::DTD(root, mode, false) {
    }

    DTD::DTD(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("DTD: DTD not implemented");
    }


    DTD::DTD(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("DTD: DTD not implemented");
    }

}
