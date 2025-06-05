#pragma once

#include "../../common.h"
using namespace std;
namespace fs = std::filesystem;


namespace xt::datasets {
    class SBDataset : public xt::datasets::Dataset{
        /*
        """`Semantic Boundaries Dataset <http://home.bharathh.info/pubs/codes/SBD/download.html>`_

    The SBD currently contains annotations from 11355 images taken from the PASCAL VOC 2011 dataset.

    .. note ::

        Please note that the train and val splits included with this dataset are different from
        the splits in the PASCAL VOC dataset. In particular some "train" images might be part of
        VOC2012 val.
        If you are interested in testing on VOC 2012 val, then use `image_set='train_noval'`,
        which excludes all val images.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (str or ``pathlib.Path``): Root directory of the Semantic Boundaries Dataset
        image_set (string, optional): Select the image_set to use, ``train``, ``val`` or ``train_noval``.
            Image set ``train_noval`` excludes VOC 2012 val images.
        mode (string, optional): Select target type. Possible values 'boundaries' or 'segmentation'.
            In case of 'boundaries', the target is an array of shape `[num_classes, H, W]`,
            where `num_classes=20`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version. Input sample is PIL image and target is a numpy array
            if `mode='boundaries'` or PIL image if `mode='segmentation'`.
    """

         */
    public :
        explicit SBDataset(const std::string& root);
        SBDataset(const std::string& root, xt::datasets::DataMode mode);
        SBDataset(const std::string& root, xt::datasets::DataMode mode, bool download);
        SBDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer);
        SBDataset(const std::string& root, xt::datasets::DataMode mode, bool download,
                   std::unique_ptr<xt::Module> transformer,
                   std::unique_ptr<xt::Module> target_transformer);


    private:
        fs::path url = fs::path("https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz");
        std::string md5 = "82b4d87ceb2ed10f6038a1cba92111cb";
        fs::path filename = fs::path("benchmark.tgz");

        fs::path voc_train_url = fs::path("https://www.cs.cornell.edu/~bharathh/train_noval.txt");
        fs::path voc_split_filename = fs::path("train_noval.txt");
        std::string voc_split_md5 = "79bff800c5f0b1ec6b21080a3c066722";

        // TODO fs::path dataset_folder_name
        fs::path dataset_folder_name = "?";

        bool download = false;
        fs::path root;
        fs::path dataset_path;

        void load_data();

        void check_resources();




    };
}
