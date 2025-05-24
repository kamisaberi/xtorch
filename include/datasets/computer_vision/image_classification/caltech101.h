#pragma once

#include "include/datasets/common.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets
{
    class Caltech101 : public xt::datasets::Dataset
    {
    public :
        explicit Caltech101(const std::string& root);
        Caltech101(const std::string& root, xt::datasets::DataMode mode);
        Caltech101(const std::string& root, xt::datasets::DataMode mode, bool download);
        Caltech101(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        Caltech101(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);

    private:
        tuple<string, string, string> resources = {
            "https://data.caltech.edu/records/mzrjq-6wc02/files/caltech-101.zip?download=1", "caltech-101.zip",
            "3138e1922a9193bfa496528edbbc45d0"
        };
        // vector<std::tuple<string, string, string> > resources = {
        //     {
        //         "https://drive.google.com/file/d/137RyRjvTBkBiIfeYBNZBtViDHQ6_Ewsp", "101_ObjectCategories.tar.gz",
        //         "b224c7392d521a49829488ab0f1120d9"
        //     },
        //     {
        //         "https://drive.google.com/file/d/175kQy3UsZ0wUEHZjqkUDdNVssr7bgh_m", "Annotations.tar",
        //         "6f83eeb1f24d99cab4eb377263132c91"
        //     }
        // };

        fs::path dataset_folder_name = "caltech-101";
        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();

        void check_resources();
    };
}
