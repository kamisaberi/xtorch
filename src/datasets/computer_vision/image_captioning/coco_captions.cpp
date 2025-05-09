#include "../../../../include/datasets/computer_vision/image_captioning/coco_captions.h"

namespace xt::data::datasets {

    // ---------------------- CocoCaptions ---------------------- //
    CocoCaptions::CocoCaptions(const std::string &root): CocoCaptions::CocoCaptions(root, DataMode::TRAIN, false) {
    }

    CocoCaptions::CocoCaptions(const std::string &root, DataMode mode): CocoCaptions::CocoCaptions(root, mode, false) {
    }

    CocoCaptions::CocoCaptions(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("CocoCaptions: CocoCaptions not implemented");
    }


    CocoCaptions::CocoCaptions(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("CocoCaptions: CocoCaptions not implemented");
    }


}
