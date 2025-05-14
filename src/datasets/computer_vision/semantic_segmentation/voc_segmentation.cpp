#include "datasets/computer_vision/semantic_segmentation/voc_segmentation.h"

namespace xt::data::datasets {


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
