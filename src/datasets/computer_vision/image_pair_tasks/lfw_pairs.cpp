#include "../../../../include/datasets/computer_vision/image_pair_tasks/lfw_pairs.h"

namespace xt::data::datasets {

    // ---------------------- LFW ---------------------- //
    LFWPairs::LFWPairs(const std::string &root): LFWPairs::LFWPairs(root, DataMode::TRAIN, false) {
    }

    LFWPairs::LFWPairs(const std::string &root, DataMode mode): LFWPairs::LFWPairs(root, mode, false) {
    }

    LFWPairs::LFWPairs(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("LFWPairs: LFWPairs not implemented");
    }


    LFWPairs::LFWPairs(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("LFWPairs: LFWPairs not implemented");
    }



}
