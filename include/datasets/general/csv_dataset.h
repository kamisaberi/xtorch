#pragma once


// #include "../base/base.h"
// #include "../../headers/datasets.h"
#include "../../../third_party/csv2/reader.hpp"
#include "datasets/common.h"



namespace xt::data::datasets {
    class CSVDataset : public xt::datasets::Dataset {
    public :
        CSVDataset(const std::string &file_path);

        CSVDataset(const std::string &file_path, DataMode mode);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, int y_index);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, vector<int> y_indices);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, string y_title);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, vector<string> y_titles);

        CSVDataset(const std::string &file_path, DataMode mode, std::unique_ptr<xt::Module> transformer);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, int y_index, std::unique_ptr<xt::Module> transformer);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, vector<int> y_indices, std::unique_ptr<xt::Module> transformer);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, string y_title, std::unique_ptr<xt::Module> transformer);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, vector<string> y_titles, std::unique_ptr<xt::Module> transformer);

    private:
        string file_path_;
        vector<int> x_indices;
        vector<int> y_indices;
        vector<string> labels_name;
        bool load_sub_folders = false;

        void load_data();
    };


    class StackedCSVDataset : public xt::datasets::Dataset {
    public :
        StackedCSVDataset(const std::string &folder_path);

        StackedCSVDataset(const std::string &folder_path, DataMode mode);

        StackedCSVDataset(const std::string &folder_path, DataMode mode, bool load_sub_folders);

        StackedCSVDataset(const std::string &folder_path, DataMode mode, bool load_sub_folders,
                          std::unique_ptr<xt::Module> transformer);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;

        void load_data();
    };
}
