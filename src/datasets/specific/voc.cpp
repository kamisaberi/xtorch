#include "../../../include/datasets/specific/voc.h"

namespace xt::data::datasets {

    // ---------------------- VOCDetection ---------------------- //
    VOCDetection::VOCDetection(const std::string &root): VOCDetection::VOCDetection(root, DataMode::TRAIN, false) {
    }

    VOCDetection::VOCDetection(const std::string &root, DataMode mode): VOCDetection::VOCDetection(root, mode, false) {
    }

    VOCDetection::VOCDetection(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("VOCDetection: VOCDetection not implemented");
    }


    VOCDetection::VOCDetection(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("VOCDetection: VOCDetection not implemented");
    }



    // ---------------------- VOCSegmentation ---------------------- //
    VOCSegmentation::VOCSegmentation(const std::string &root): VOCSegmentation::VOCSegmentation(root, DataMode::TRAIN, false) {
    }

    VOCSegmentation::VOCSegmentation(const std::string &root, DataMode mode): VOCSegmentation::VOCSegmentation(root, mode, false) {
    }

    VOCSegmentation::VOCSegmentation(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("VOCSegmentation: VOCSegmentation not implemented");
    }


    VOCSegmentation::VOCSegmentation(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("VOCSegmentation: VOCSegmentation not implemented");
    }

}
