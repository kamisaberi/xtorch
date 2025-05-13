#include "../../../../include/datasets/computer_vision/image_classification/lfw_people.h"

namespace xt::data::datasets {

    // ---------------------- LFW ---------------------- //
    LFWPeople::LFWPeople(const std::string &root): LFWPeople::LFWPeople(root, DataMode::TRAIN, false) {
    }

    LFWPeople::LFWPeople(const std::string &root, DataMode mode): LFWPeople::LFWPeople(root, mode, false) {
    }

    LFWPeople::LFWPeople(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("LFW: LFW not implemented");
    }


    LFWPeople::LFWPeople(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("LFW: LFW not implemented");
    }


}
