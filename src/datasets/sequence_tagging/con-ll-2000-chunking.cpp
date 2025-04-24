#include "../../../include/datasets/sequence-tagging/con-ll-2000-chunking.h"

namespace xt::data::datasets {

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string &root): CoNLL2000Chunking::CoNLL2000Chunking(root, DataMode::TRAIN, false) {
    }

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string &root, DataMode mode): CoNLL2000Chunking::CoNLL2000Chunking(root, mode, false) {
    }

    CoNLL2000Chunking::CoNLL2000Chunking(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CoNLL2000Chunking: CoNLL2000Chunking not implemented");
    }


    CoNLL2000Chunking::CoNLL2000Chunking(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CoNLL2000Chunking: CoNLL2000Chunking not implemented");
    }


}
