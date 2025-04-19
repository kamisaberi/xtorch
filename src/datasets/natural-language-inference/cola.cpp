#include "../../../include/datasets/natural-language-inference/cola.h"

namespace xt::data::datasets {

    COLA::COLA(const std::string &root): COLA::COLA(root, DataMode::TRAIN, false) {
    }

    COLA::COLA(const std::string &root, DataMode mode): COLA::COLA(root, mode, false) {
    }

    COLA::COLA(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("COLA: COLA not implemented");
    }


    COLA::COLA(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("COLA: COLA not implemented");
    }


}
