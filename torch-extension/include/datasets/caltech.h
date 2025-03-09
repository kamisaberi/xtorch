#pragma once

#include "../base/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class Caltech101 : public BaseDataset {
    public :
        Caltech101(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Caltech101(const fs::path &root, DatasetArguments args);

    private:
        vector<std::tuple<string, string, string> > resources = {
            {
                "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp", "101_ObjectCategories.tar.gz",
                "b224c7392d521a49829488ab0f1120d9"
            },
            {
                "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m", "Annotations.tar",
                "6f83eeb1f24d99cab4eb377263132c91"
            }
        };

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class Caltech256 : public BaseDataset {
    public :
        Caltech256(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        Caltech256(const fs::path &root, DatasetArguments args);

    private:
        vector<std::tuple<string, string, string> > resources = {
            {
                "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK", "256_ObjectCategories.tar",
                "67b4f42ca05d46448c6bb8ecd2220f6d"
            },
        };

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
