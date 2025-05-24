#pragma once


#include "include/datasets/common.h"

namespace xt::datasets {



    class VideoDataset : public xt::datasets::Dataset {
    public :
        VideoDataset(const std::string &file_path);
        VideoDataset(const std::string &file_path,DataMode mode);
        VideoDataset(const std::string &file_path,DataMode mode , std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };


    class StackedVideoDataset : public xt::datasets::Dataset {
    public :
        StackedVideoDataset(const std::string &folder_path);
        StackedVideoDataset(const std::string &folder_path,DataMode mode);
        StackedVideoDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders);
        StackedVideoDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders , std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
