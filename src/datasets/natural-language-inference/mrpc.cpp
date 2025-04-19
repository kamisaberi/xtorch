#include "../../../include/datasets/natural-language-inference/mrpc.h"

namespace xt::data::datasets {

    MRPC::MRPC(const std::string &root): MRPC::MRPC(root, DataMode::TRAIN, false) {
    }

    MRPC::MRPC(const std::string &root, DataMode mode): MRPC::MRPC(root, mode, false) {
    }

    MRPC::MRPC(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("MRPC: MRPC not implemented");
    }


    MRPC::MRPC(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("MRPC: MRPC not implemented");
    }


}
