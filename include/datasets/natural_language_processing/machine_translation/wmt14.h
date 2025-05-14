#pragma once
#include "../../base/base.h"
#include "../../../headers/datasets.h"


namespace xt::data::datasets {
    class WMT14 : BaseDataset {
        public :
            explicit WMT14(const std::string &root);
        WMT14(const std::string &root, DataMode mode);
        WMT14(const std::string &root, DataMode mode , bool download);
        WMT14(const std::string &root, DataMode mode , bool download, TransformType transforms);

        private :
            void load_data();

        void check_resources();
    };
}
