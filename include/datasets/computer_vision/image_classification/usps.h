#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets {
    class USPS : xt::datasets::Dataset {
        /*
        """`USPS <https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html#usps>`_ Dataset.
    The data-format is : [label [index:value ]*256 \\n] * num_lines, where ``label`` lies in ``[1, 10]``.
    The value for each pixel lies in ``[-1, 1]``. Here we transform the ``label`` into ``[0, 9]``
    and make pixel values in ``[0, 255]``.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset to store``USPS`` data files.
        train (bool, optional): If True, creates dataset from ``usps.bz2``,
            otherwise from ``usps.t.bz2``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

         */
    public :

        explicit USPS(const std::string& root);
        USPS(const std::string& root, xt::datasets::DataMode mode);
        USPS(const std::string& root, xt::datasets::DataMode mode, bool download);
        USPS(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        USPS(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private :
        std::map<std::string, std::tuple<fs::path, fs::path, std::string>> resources = {
                {"train", {
                                  fs::path(
                                          "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2"),
                                  fs::path("usps.bz2"),
                                  "ec16c51db3855ca6c91edd34d0e9b197"
                          }

                },
                {"test",  {
                                  fs::path(
                                          "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2"),
                                  fs::path("usps.t.bz2"),
                                  "8ea070ee2aca1ac39742fdd1ef5ed118"
                          }
                }
        };


        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();


    };
}
