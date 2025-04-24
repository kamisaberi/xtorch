#include "../../../include/datasets/image_classification/fer.h"

namespace xt::data::datasets {

    FER2013::FER2013(const std::string &root): FER2013::FER2013(root, DataMode::TRAIN, false) {
    }

    FER2013::FER2013(const std::string &root, DataMode mode): FER2013::FER2013(root, mode, false) {
    }

    FER2013::FER2013(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("FER2013: FER2013 not implemented");
    }


    FER2013::FER2013(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("FER2013: FER2013 not implemented");
    }

}
