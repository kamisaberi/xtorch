#include "../../../include/datasets/image_classification/semeion.h"

namespace xt::data::datasets {

    SEMEION::SEMEION(const std::string &root): SEMEION::SEMEION(root, DataMode::TRAIN, false) {
    }

    SEMEION::SEMEION(const std::string &root, DataMode mode): SEMEION::SEMEION(root, mode, false) {
    }

    SEMEION::SEMEION(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SEMEION: SEMEION not implemented");
    }


    SEMEION::SEMEION(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SEMEION: SEMEION not implemented");
    }


}
