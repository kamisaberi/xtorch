#include "../../../include/datasets/machine_translation/iwslt.h"

namespace xt::data::datasets {

    IWSLT::IWSLT(const std::string &root): IWSLT::IWSLT(root, DataMode::TRAIN, false) {
    }

    IWSLT::IWSLT(const std::string &root, DataMode mode): IWSLT::IWSLT(root, mode, false) {
    }

    IWSLT::IWSLT(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("IWSLT: IWSLT not implemented");
    }


    IWSLT::IWSLT(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("IWSLT: IWSLT not implemented");
    }


}
