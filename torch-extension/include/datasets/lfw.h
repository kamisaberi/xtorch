#pragma once
#include "base.h"
#include "../headers/datasets.h"


using namespace std;
namespace fs = std::filesystem;

namespace xt::data::datasets {
    class LFW : public BaseDataset {
    public :
        LFW(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        LFW(const fs::path &root, DatasetArguments args);

    private :
        fs::path dataset_folder_name = fs::path("lfw-py");
        fs::path url = fs::path("http://vis-www.cs.umass.edu/lfw/");
        std::map<std::string, std::tuple<std::string, fs::path, std::string> > file_dict = {
            {"original", {"lfw", fs::path("lfw.tgz"), "a17d05bd522c52d84eca14327a23d494"}},
            {"funneled", {"lfw_funneled", fs::path("lfw-funneled.tgz"), "1b42dfed7d15c9b2dd63d5e5840c86ad"}},
            {
                "deepfunneled",
                {"lfw-deepfunneled", fs::path("lfw-deepfunneled.tgz"), "68331da3eb755a505a502b5aacb3c201"}
            }
        };

        std::map<std::string, std::string> checksums = {
            {"pairs.txt", "9f1ba174e4e1c508ff7cdf10ac338a7d"},
            {"pairsDevTest.txt", "5132f7440eb68cf58910c8a45a2ac10b"},
            {"pairsDevTrain.txt", "4f27cbf15b2da4a85c1907eb4181ad21"},
            {"people.txt", "450f0863dd89e85e73936a6d71a3474b"},
            {"peopleDevTest.txt", "e4bf5be0a43b5dcd9dc5ccfcb8fb19c5"},
            {"peopleDevTrain.txt", "54eaac34beb6d042ed3a7d883e247a21"},
            {"lfw-names.txt", "a6d0a479bd074669f656265a6e693f6d"}
        };

        std::map<std::string, std::string> annot_file = {
            {"10fold", ""},
            {"train", "DevTrain"},
            {"test", "DevTest"}
        };
        fs::path names = fs::path("lfw-names.txt");

        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class LFWPeople : public LFW {
    public :
        LFWPeople(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        LFWPeople(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };

    class LFWPairs : public LFW {
    public :
        LFWPairs(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        LFWPairs(const fs::path &root, DatasetArguments args);

    private :
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
