#pragma once

#include "include/datasets/common.h"

using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
   class SEMEION : xt::datasets::Dataset {
       /*
       r"""`SEMEION <http://archive.ics.uci.edu/ml/datasets/semeion+handwritten+digit>`_ Dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory of dataset where directory
            ``semeion.py`` exists.
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

       explicit SEMEION(const std::string& root);
       SEMEION(const std::string& root, xt::datasets::DataMode mode);
       SEMEION(const std::string& root, xt::datasets::DataMode mode, bool download);
       SEMEION(const std::string& root, xt::datasets::DataMode mode, bool download,
                  std::unique_ptr<xt::Module> transformer);
       SEMEION(const std::string& root, xt::datasets::DataMode mode, bool download,
                  std::unique_ptr<xt::Module> transformer,
                  std::unique_ptr<xt::Module> target_transformer);


   private:
       fs::path url = fs::path("http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data");
       fs::path dataset_file_name = fs::path("semeion.data");
       std::string dataset_file_md5 = "cb545d371d2ce14ec121470795a77432";
       fs::path dataset_folder_name = "semeion";

       bool download = false;
       fs::path root;
       fs::path dataset_path;

       void load_data();

       void check_resources();


    };
}
