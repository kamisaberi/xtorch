#include "../../../include/datasets/natural-language-inference/snli.h"

namespace xt::data::datasets {

    SNLI::SNLI(const std::string &root): SNLI::SNLI(root, DataMode::TRAIN, false) {
    }

    SNLI::SNLI(const std::string &root, DataMode mode): SNLI::SNLI(root, mode, false) {
    }

    SNLI::SNLI(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("SNLI: SNLI not implemented");
    }


    SNLI::SNLI(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("SNLI: SNLI not implemented");
    }


}
