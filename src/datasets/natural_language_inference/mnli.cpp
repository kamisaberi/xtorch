#include "../../../include/datasets/natural-language-inference/mnli.h"

namespace xt::data::datasets {

    MNLI::MNLI(const std::string &root): MNLI::MNLI(root, DataMode::TRAIN, false) {
    }

    MNLI::MNLI(const std::string &root, DataMode mode): MNLI::MNLI(root, mode, false) {
    }

    MNLI::MNLI(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("MNLI: MNLI not implemented");
    }


    MNLI::MNLI(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("MNLI: MNLI not implemented");
    }


}
