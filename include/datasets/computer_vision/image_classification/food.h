#pragma once

#include "../../common.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::datasets
{
    class Food101 : public xt::datasets::Dataset
    {
    public:
        explicit Food101(const std::string& root);
        Food101(const std::string& root, xt::datasets::DataMode mode);
        Food101(const std::string& root, xt::datasets::DataMode mode, bool download);
        Food101(const std::string& root, xt::datasets::DataMode mode, bool download,
                std::unique_ptr<xt::Module> transformer);
        Food101(const std::string& root, xt::datasets::DataMode mode, bool download,
                std::unique_ptr<xt::Module> transformer,
                std::unique_ptr<xt::Module> target_transformer);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;

    private:
        std::string url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz";
        vector<fs::path> files ;

        fs::path dataset_file_name = "food-101.tar.gz";

        std::string dataset_file_md5 = "85eeb15f3717b99a5da872d97d918f87";

        fs::path dataset_folder_name = "food-101";

        std::size_t images_number = 101'000;

        std::vector<string> classes_name;

        std::map<string, int> classes_map;

        bool download = false;
        fs::path root;
        fs::path dataset_path;


        void load_data();

        void load_classes();

        void check_resources();
    };
}
