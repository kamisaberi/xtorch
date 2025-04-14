#include "../../../include/datasets/image-classification/stl.h"

namespace xt::data::datasets {

    STL10::STL10(const std::string &root): STL10::STL10(root, DataMode::TRAIN, false) {
    }

    STL10::STL10(const std::string &root, DataMode mode): STL10::STL10(root, mode, false) {
    }

    STL10::STL10(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("STL10: STL10 not implemented");
    }


    STL10::STL10(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("STL10: STL10 not implemented");
    }


}
