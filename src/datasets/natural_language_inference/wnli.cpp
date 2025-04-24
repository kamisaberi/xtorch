#include "../../../include/datasets/natural_language_inference/wnli.h"

namespace xt::data::datasets {

    WNLI::WNLI(const std::string &root): WNLI::WNLI(root, DataMode::TRAIN, false) {
    }

    WNLI::WNLI(const std::string &root, DataMode mode): WNLI::WNLI(root, mode, false) {
    }

    WNLI::WNLI(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("WNLI: WNLI not implemented");
    }


    WNLI::WNLI(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("WNLI: WNLI not implemented");
    }


}
