#include "../../../include/datasets/sequence_tagging/con_ll_2000_chunking.h"

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
