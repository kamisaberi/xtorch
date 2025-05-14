#pragma once


#include "datasets/base/base.h"

namespace xt::data::datasets {
    class CarlaStereo : public  BaseDataset {
    public :
        explicit  CarlaStereo(const std::string &root);
        CarlaStereo(const std::string &root, DataMode mode);
        CarlaStereo(const std::string &root, DataMode mode , bool download);
        CarlaStereo(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private :
        void load_data();

        void check_resources();
    };



}
