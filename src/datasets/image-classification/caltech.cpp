#include "../../../include/datasets/image-classification/caltech.h"

namespace xt::data::datasets {


    // ---------------------- Caltech101 ---------------------- //

    Caltech101::Caltech101(const std::string &root): Caltech101::Caltech101(root, DataMode::TRAIN, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode): Caltech101::Caltech101(root, mode, false) {
    }

    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }


    Caltech101::Caltech101(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }



    // ---------------------- Caltech256 ---------------------- //

    Caltech256::Caltech256(const std::string &root): Caltech256::Caltech256(root, DataMode::TRAIN, false) {
    }

    Caltech256::Caltech256(const std::string &root, DataMode mode): Caltech256::Caltech256(root, mode, false) {
    }

    Caltech256::Caltech256(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Caltech256: Caltech256 not implemented");
    }


    Caltech256::Caltech256(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Caltech101: Caltech101 not implemented");
    }


}
