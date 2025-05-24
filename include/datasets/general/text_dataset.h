#pragma once


#include "include/datasets/common.h"

namespace xt::datasets {



    class TextDataset : public xt::datasets::Dataset {
    public :
        TextDataset(const std::string &file_path);
        TextDataset(const std::string &file_path,DataMode mode);
        TextDataset(const std::string &file_path,DataMode mode , std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };


    class StackedTextDataset : public xt::datasets::Dataset {
    public :
        StackedTextDataset(const std::string &folder_path);
        StackedTextDataset(const std::string &folder_path,DataMode mode);
        StackedTextDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders);
        StackedTextDataset(const std::string &folder_path,DataMode mode, bool load_sub_folders , std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;
        void load_data();
    };



}
