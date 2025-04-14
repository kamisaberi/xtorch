#include "../../../include/datasets/video/ucf.h"

namespace xt::data::datasets {


    UCF101::UCF101(const std::string &root): UCF101::UCF101(root, DataMode::TRAIN, false) {
    }

    UCF101::UCF101(const std::string &root, DataMode mode): UCF101::UCF101(root, mode, false) {
    }

    UCF101::UCF101(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("UCF101: UCF101 not implemented");
    }


    UCF101::UCF101(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("UCF101: UCF101 not implemented");
    }



}
