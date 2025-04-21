

# File csv-dataset.h

[**File List**](files.md) **>** [**datasets**](dir_29ff4802398ba4a572b958e731c7adb4.md) **>** [**general**](dir_3e490c73b2bbc01f3b90ef3b6e284c64.md) **>** [**csv-dataset.h**](csv-dataset_8h.md)

[Go to the documentation of this file](csv-dataset_8h.md)


```C++
#pragma once


#include "../base/base.h"
#include "../../headers/datasets.h"
#include "../../../third_party/csv2/reader.hpp"



namespace xt::data::datasets {
    class CSVDataset : public BaseDataset {
    public :
        CSVDataset(const std::string &file_path);

        CSVDataset(const std::string &file_path, DataMode mode);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, int y_index);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, vector<int> y_indices);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, string y_title);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, vector<string> y_titles);

        CSVDataset(const std::string &file_path, DataMode mode, TransformType transforms);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, int y_index, TransformType transforms);

        CSVDataset(const std::string &file_path, DataMode mode, vector<int> x_indices, vector<int> y_indices, TransformType transforms);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, string y_title, TransformType transforms);

        CSVDataset(const std::string &file_path, DataMode mode, vector<string> x_titles, vector<string> y_titles, TransformType transforms);

    private:
        vector<int> x_indices;
        vector<int> y_indices;
        vector<string> labels_name;
        bool load_sub_folders = false;

        void load_data();
    };


    class StackedCSVDataset : public BaseDataset {
    public :
        StackedCSVDataset(const std::string &folder_path);

        StackedCSVDataset(const std::string &folder_path, DataMode mode);

        StackedCSVDataset(const std::string &folder_path, DataMode mode, bool load_sub_folders);

        StackedCSVDataset(const std::string &folder_path, DataMode mode, bool load_sub_folders,
                          vector<std::function<torch::Tensor(torch::Tensor)> > transforms);

    private:
        vector<string> labels_name;
        bool load_sub_folders = false;

        void load_data();
    };
}
```


