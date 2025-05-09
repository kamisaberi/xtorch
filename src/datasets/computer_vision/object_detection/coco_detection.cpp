#include "../../../include/datasets/object_detection_and_segmentation/coco.h"

namespace xt::data::datasets {

    // ---------------------- CocoDetection ---------------------- //
    CocoDetection::CocoDetection(const std::string &root): CocoDetection::CocoDetection(root, DataMode::TRAIN, false) {
    }

    CocoDetection::CocoDetection(const std::string &root, DataMode mode): CocoDetection::CocoDetection(root, mode, false) {
    }

    CocoDetection::CocoDetection(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CocoDetection: CocoDetection not implemented");
    }


    CocoDetection::CocoDetection(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CocoDetection: CocoDetection not implemented");
    }




}
