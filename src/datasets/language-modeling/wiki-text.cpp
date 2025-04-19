#include "../../../include/datasets/language-modeling/wiki-text.h"

namespace xt::data::datasets {

    WikiText::WikiText(const std::string &root): WikiText::WikiText(root, DataMode::TRAIN, false) {
    }

    WikiText::WikiText(const std::string &root, DataMode mode): WikiText::WikiText(root, mode, false) {
    }

    WikiText::WikiText(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("WikiText: WikiText not implemented");
    }


    WikiText::WikiText(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("WikiText: WikiText not implemented");
    }


}
