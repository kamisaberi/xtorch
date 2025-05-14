#pragma once

#include "datasets/base/base.h"
#include "datasets/common.h"


namespace xt::data::datasets {
    class LjSpeech : BaseDataset {
    public :
        explicit LjSpeech(const std::string &root);

        LjSpeech(const std::string &root, DataMode mode);

        LjSpeech(const std::string &root, DataMode mode, bool download);

        LjSpeech(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
