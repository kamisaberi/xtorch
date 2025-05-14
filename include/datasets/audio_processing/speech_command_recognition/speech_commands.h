#pragma once

#include "datasets/base/base.h"
#include "datasets/common.h"


namespace xt::data::datasets {
    class SpeechCommands : BaseDataset {
    public :
        explicit SpeechCommands(const std::string &root);

        SpeechCommands(const std::string &root, DataMode mode);

        SpeechCommands(const std::string &root, DataMode mode, bool download);

        SpeechCommands(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
