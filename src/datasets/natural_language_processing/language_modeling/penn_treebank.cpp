#include "../../../include/datasets/language_modeling/penn_treebank.h"

namespace xt::data::datasets {

    PennTreebank::PennTreebank(const std::string &root): PennTreebank::PennTreebank(root, DataMode::TRAIN, false) {
    }

    PennTreebank::PennTreebank(const std::string &root, DataMode mode): PennTreebank::PennTreebank(root, mode, false) {
    }

    PennTreebank::PennTreebank(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("PennTreebank: PennTreebank not implemented");
    }


    PennTreebank::PennTreebank(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("PennTreebank: PennTreebank not implemented");
    }


}
