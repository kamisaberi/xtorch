#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class DTD : public BaseDataset {
    public :
        explicit  DTD(const std::string &root);
        DTD(const std::string &root, DataMode mode);
        DTD(const std::string &root, DataMode mode , bool download);
        DTD(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
