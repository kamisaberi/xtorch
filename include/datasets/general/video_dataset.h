#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {



    class VideoDataset : public BaseDataset {
    public :
        VideoDataset(const std::string &file_path);
        VideoDataset(const std::string &file_path,DataMode mode);
        VideoDataset(const std::string &file_path,DataMode mode , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };


    class StackedVideoDataset : public BaseDataset {
    public :
        StackedVideoDataset(const std::string &folder_path);
        StackedVideoDataset(const std::string &folder_path,DataMode mode);
        StackedVideoDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders);
        StackedVideoDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders , vector<std::function<torch::Tensor(torch::Tensor)>> transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
