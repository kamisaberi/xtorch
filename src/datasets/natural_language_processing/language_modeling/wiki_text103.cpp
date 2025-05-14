#include "datasets/natural_language_processing/language_modeling/wiki_text103.h"

namespace xt::data::datasets {

    WikiText103::WikiText103(const std::string &root): WikiText103::WikiText103(root, DataMode::TRAIN, false) {
    }

    WikiText103::WikiText103(const std::string &root, DataMode mode): WikiText103::WikiText103(root, mode, false) {
    }

    WikiText103::WikiText103(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("WikiText: WikiText not implemented");
    }


    WikiText103::WikiText103(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("WikiText: WikiText not implemented");
    }


}
