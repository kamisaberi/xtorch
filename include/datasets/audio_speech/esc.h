#pragma once

#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class ESC : BaseDataset {
    public :
        explicit ESC(const std::string &root);

        ESC(const std::string &root, DataMode mode);

        ESC(const std::string &root, DataMode mode, bool download);

        ESC(const std::string &root, DataMode mode, bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
