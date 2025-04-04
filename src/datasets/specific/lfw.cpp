#include "../../../include/datasets/specific/lfw.h"

namespace xt::data::datasets {

    // ---------------------- LFW ---------------------- //
    LFW::LFW(const std::string &root): LFW::LFW(root, DataMode::TRAIN, false) {
    }

    LFW::LFW(const std::string &root, DataMode mode): LFW::LFW(root, mode, false) {
    }

    LFW::LFW(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("LFW: LFW not implemented");
    }


    LFW::LFW(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("LFW: LFW not implemented");
    }



    // ---------------------- LFWPeople ---------------------- //
    LFWPeople::LFWPeople(const std::string &root): LFWPeople::LFWPeople(root, DataMode::TRAIN, false) {
    }

    LFWPeople::LFWPeople(const std::string &root, DataMode mode): LFWPeople::LFWPeople(root, mode, false) {
    }

    LFWPeople::LFWPeople(const std::string &root, DataMode mode, bool download) : LFW(root, mode, download) {
        throw std::runtime_error("LFWPeople: LFWPeople not implemented");
    }


    LFWPeople::LFWPeople(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : LFW(root, mode, download, transforms) {
        throw std::runtime_error("LFWPeople: LFWPeople not implemented");
    }



    // ---------------------- LFWPairs ---------------------- //
    LFWPairs::LFWPairs(const std::string &root): LFWPairs::LFWPairs(root, DataMode::TRAIN, false) {
    }

    LFWPairs::LFWPairs(const std::string &root, DataMode mode): LFWPairs::LFWPairs(root, mode, false) {
    }

    LFWPairs::LFWPairs(const std::string &root, DataMode mode, bool download) : LFW(root, mode, download) {
        throw std::runtime_error("LFWPairs: LFWPairs not implemented");
    }


    LFWPairs::LFWPairs(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : LFW(root, mode, download, transforms) {
        throw std::runtime_error("LFWPairs: LFWPairs not implemented");
    }


}
