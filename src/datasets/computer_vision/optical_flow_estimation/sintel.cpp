#include "../../../include/datasets/specific/sintel.h"

namespace xt::data::datasets {

    // ---------------------- Sintel ---------------------- //

    Sintel::Sintel(const std::string &root): Sintel::Sintel(root, DataMode::TRAIN, false) {
    }

    Sintel::Sintel(const std::string &root, DataMode mode): Sintel::Sintel(root, mode, false) {
    }

    Sintel::Sintel(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("Sintel: Sintel not implemented");
    }


    Sintel::Sintel(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("Sintel: Sintel not implemented");
    }



}
