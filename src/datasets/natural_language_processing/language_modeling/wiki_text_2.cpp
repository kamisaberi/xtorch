#include "datasets/natural_language_processing/language_modeling/wiki_text_2.h"

namespace xt::data::datasets {

    WikiTextV2::WikiTextV2(const std::string &root): WikiTextV2::WikiTextV2(root, DataMode::TRAIN, false) {
    }

    WikiTextV2::WikiTextV2(const std::string &root, DataMode mode): WikiTextV2::WikiTextV2(root, mode, false) {
    }

    WikiTextV2::WikiTextV2(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("WikiText: WikiText not implemented");
    }


    WikiTextV2::WikiTextV2(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("WikiText: WikiText not implemented");
    }


}
