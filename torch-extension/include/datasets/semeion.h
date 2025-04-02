#pragma once

#include "../headers/datasets.h"
#include "base.h"


using namespace std;
namespace fs = std::filesystem;


namespace xt::data::datasets {
   class SEMEION : BaseDataset {
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
       SEMEION(const std::string &root, DataMode mode = DataMode::TRAIN, bool download = false);

       SEMEION(const fs::path &root, DatasetArguments args);

   private:
       fs::path url = fs::path("http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data");
       fs::path dataset_file_name = fs::path("semeion.data");
       std::string dataset_file_md5 = "cb545d371d2ce14ec121470795a77432";
       fs::path dataset_folder_name = "semeion";

       void load_data(DataMode mode = DataMode::TRAIN);

       void check_resources(const std::string &root, bool download = false);

    };
}
