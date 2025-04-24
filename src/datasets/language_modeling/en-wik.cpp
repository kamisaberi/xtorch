#include "../../../include/datasets/language-modeling/en-wik.h"

namespace xt::data::datasets {

    EnWik::EnWik(const std::string &root): EnWik::EnWik(root, DataMode::TRAIN, false) {
    }

    EnWik::EnWik(const std::string &root, DataMode mode): EnWik::EnWik(root, mode, false) {
    }

    EnWik::EnWik(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("EnWik: EnWik not implemented");
    }


    EnWik::EnWik(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("EnWik: EnWik not implemented");
    }


}
