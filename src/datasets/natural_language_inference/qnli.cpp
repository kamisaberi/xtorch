#include "../../../include/datasets/natural-language-inference/qnli.h"

namespace xt::data::datasets {

    QNLI::QNLI(const std::string &root): QNLI::QNLI(root, DataMode::TRAIN, false) {
    }

    QNLI::QNLI(const std::string &root, DataMode mode): QNLI::QNLI(root, mode, false) {
    }

    QNLI::QNLI(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("QNLI: QNLI not implemented");
    }


    QNLI::QNLI(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("QNLI: QNLI not implemented");
    }


}
