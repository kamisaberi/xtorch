#pragma once
#include "../../../third_party/csv2/reader.hpp"
#include "../common.h"




namespace xt::datasets {
    class CSVFolder : public xt::datasets::Dataset {
    public :
        CSVFolder(const std::string &file_path);

        CSVFolder(const std::string &file_path, DataMode mode);

        CSVFolder(const std::string &file_path, DataMode mode, vector<int> x_indices, int y_index);

        CSVFolder(const std::string &file_path, DataMode mode, vector<int> x_indices, vector<int> y_indices);

        CSVFolder(const std::string &file_path, DataMode mode, vector<string> x_titles, string y_title);

        CSVFolder(const std::string &file_path, DataMode mode, vector<string> x_titles, vector<string> y_titles);

        CSVFolder(const std::string &file_path, DataMode mode, std::unique_ptr<xt::Module> transformer);

        CSVFolder(const std::string &file_path, DataMode mode, vector<int> x_indices, int y_index, std::unique_ptr<xt::Module> transformer);

        CSVFolder(const std::string &file_path, DataMode mode, vector<int> x_indices, vector<int> y_indices, std::unique_ptr<xt::Module> transformer);

        CSVFolder(const std::string &file_path, DataMode mode, vector<string> x_titles, string y_title, std::unique_ptr<xt::Module> transformer);

        CSVFolder(const std::string &file_path, DataMode mode, vector<string> x_titles, vector<string> y_titles, std::unique_ptr<xt::Module> transformer);

    private:
        string file_path_;
        vector<int> x_indices;
        vector<int> y_indices;
        vector<string> labels_name;
        bool load_sub_folders = false;

        void load_data();
    };


}
