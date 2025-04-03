#include "../../../include/datasets/specific/wider-face.h"

namespace xt::data::datasets {

    WIDERFace::WIDERFace(const std::string &root): WIDERFace::WIDERFace(root, DataMode::TRAIN, false) {
    }

    WIDERFace::WIDERFace(const std::string &root, DataMode mode): WIDERFace::WIDERFace(root, mode, false) {
    }

    WIDERFace::WIDERFace(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("WIDERFace: WIDERFace not implemented");
    }


    WIDERFace::WIDERFace(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("WIDERFace: WIDERFace not implemented");
    }


}
