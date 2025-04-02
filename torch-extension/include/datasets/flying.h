#pragma once

#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class FlyingChairs : public BaseDataset {
    public :
        FlyingChairs(const std::string &root);
        FlyingChairs(const std::string &root, DataMode mode);
        FlyingChairs(const std::string &root, DataMode mode , bool download);
        FlyingChairs(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class FlyingThings3D : public BaseDataset {
    public :
        FlyingThings3D(const std::string &root);
        FlyingThings3D(const std::string &root, DataMode mode);
        FlyingThings3D(const std::string &root, DataMode mode , bool download);
        FlyingThings3D(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
