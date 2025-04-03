#pragma once
#include "../headers/datasets.h"
#include "base.h"


namespace xt::data::datasets {
    class DTD : public BaseDataset {
    public :
        DTD(const std::string &root);
        DTD(const std::string &root, DataMode mode);
        DTD(const std::string &root, DataMode mode , bool download);
        DTD(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :
        void load_data();

        void check_resources();
    };
}
