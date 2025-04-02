#pragma once

#include "../headers/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class Caltech101 : public BaseDataset {
    public :
        Caltech101(const std::string &root);
        Caltech101(const std::string &root, DataMode mode);
        Caltech101(const std::string &root, DataMode mode , bool download);
        Caltech101(const std::string &root, DataMode mode , bool download, TransformType transforms);

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
        fs::path dataset_folder_name = "caltech101";
        void load_data();

        void check_resources();
    };

    class Caltech256 : public BaseDataset {
    public :
        Caltech256(const std::string &root);
        Caltech256(const std::string &root, DataMode mode);
        Caltech256(const std::string &root, DataMode mode , bool download);
        Caltech256(const std::string &root, DataMode mode , bool download, TransformType transforms);


    private:
        vector<std::tuple<string, string, string> > resources = {
            {
                "https://drive.google.com/file/d/1r6o0pSROcV1_VwT4oSjA2FBUSCWGuxLK", "256_ObjectCategories.tar",
                "67b4f42ca05d46448c6bb8ecd2220f6d"
            },
        };
        fs::path dataset_folder_name = "caltech256";

        void load_data();

        void check_resources();
    };
}
