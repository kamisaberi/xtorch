#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class Middlebury2014Stereo : BaseDataset {
        /*
             """Publicly available scenes from the Middlebury dataset `2014 version <https://vision.middlebury.edu/stereo/data/scenes2014/>`.

    The dataset mostly follows the original format, without containing the ambient subdirectories.  : ::

        root
            Middlebury2014
                train
                    scene1-{perfect,imperfect}
                        calib.txt
                        im{0,1}.png
                        im1E.png
                        im1L.png
                        disp{0,1}.pfm
                        disp{0,1}-n.png
                        disp{0,1}-sd.pfm
                        disp{0,1}y.pfm
                    scene2-{perfect,imperfect}
                        calib.txt
                        im{0,1}.png
                        im1E.png
                        im1L.png
                        disp{0,1}.pfm
                        disp{0,1}-n.png
                        disp{0,1}-sd.pfm
                        disp{0,1}y.pfm
                    ...
                additional
                    scene1-{perfect,imperfect}
                        calib.txt
                        im{0,1}.png
                        im1E.png
                        im1L.png
                        disp{0,1}.pfm
                        disp{0,1}-n.png
                        disp{0,1}-sd.pfm
                        disp{0,1}y.pfm
                    ...
                test
                    scene1
                        calib.txt
                        im{0,1}.png
                    scene2
                        calib.txt
                        im{0,1}.png
                    ...

    Args:
        root (str or ``pathlib.Path``): Root directory of the Middleburry 2014 Dataset.
        split (string, optional): The dataset split of scenes, either "train" (default), "test", or "additional"
        use_ambient_views (boolean, optional): Whether to use different expose or lightning views when possible.
            The dataset samples with equal probability between ``[im1.png, im1E.png, im1L.png]``.
        calibration (string, optional): Whether or not to use the calibrated (default) or uncalibrated scenes.
        transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
        download (boolean, optional): Whether or not to download the dataset in the ``root`` directory.
    """

         */
    public :
        Middlebury2014Stereo(const std::string &root);
        Middlebury2014Stereo(const std::string &root, DataMode mode);
        Middlebury2014Stereo(const std::string &root, DataMode mode , bool download);
        Middlebury2014Stereo(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        fs::path dataset_folder_name = fs::path("Middlebury2014");
        void load_data();

        void check_resources();
    };
}
