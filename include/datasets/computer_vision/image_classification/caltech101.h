#pragma once

#include "datasets/base/base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
    class Caltech101 : public BaseDataset {
    public :
        explicit Caltech101(const std::string &root);
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


}
