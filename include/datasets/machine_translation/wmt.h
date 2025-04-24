#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class WMT : BaseDataset {
        public :
            explicit WMT(const std::string &root);
        WMT(const std::string &root, DataMode mode);
        WMT(const std::string &root, DataMode mode , bool download);
        WMT(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
