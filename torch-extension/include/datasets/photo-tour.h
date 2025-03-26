#pragma once

#include "../headers/datasets.h"
#include "base.h"


namespace torch::ext::data::datasets {
    class PhotoTour : BaseDataset {
    public :
        PhotoTour(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

        PhotoTour(const fs::path &root, DatasetArguments args);

    private :
        std::map<std::string, std::tuple<fs::path, fs::path, std::string> > resources = {
            {
                "notredame_harris", {
                    fs::path("http://matthewalunbrown.com/patchdata/notredame_harris.zip"), fs::path(
                        "notredame_harris.zip"),
                    "69f8c90f78e171349abdf0307afefe4d"
                }
            },
            {
                "yosemite_harris", {
                    fs::path("http://matthewalunbrown.com/patchdata/yosemite_harris.zip"), fs::path(
                        "yosemite_harris.zip"),
                    "a73253d1c6fbd3ba2613c45065c00d46"
                }
            },
            {
                "liberty_harris", {
                    fs::path("http://matthewalunbrown.com/patchdata/liberty_harris.zip"), fs::path(
                        "liberty_harris.zip"),
                    "c731fcfb3abb4091110d0ae8c7ba182c"
                }
            },
            {
                "notredame", {
                    fs::path("http://icvl.ee.ic.ac.uk/vbalnt/notredame.zip"), fs::path(
                        "notredame.zip"),
                    "509eda8535847b8c0a90bbb210c83484"
                }
            },
            {
                "yosemite", {
                    fs::path("http://icvl.ee.ic.ac.uk/vbalnt/yosemite.zip"), fs::path(
                        "yosemite.zip"),
                    "533b2e8eb7ede31be40abc317b2fd4f0"
                }
            },
            {
                "liberty", {
                    fs::path("http://icvl.ee.ic.ac.uk/vbalnt/liberty.zip"), fs::path(
                        "liberty.zip"),
                    "fdd9152f138ea5ef2091746689176414"
                }
            },


        };

        fs::path dataset_folder_name = "photo-tour";
        void load_data(DataMode mode = DataMode::TRAIN);

        void check_resources(const std::string &root, bool download = false);
    };
}
