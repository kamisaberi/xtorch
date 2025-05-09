#pragma once
#include "../../base/base.h"
#include "../../../headers/datasets.h"


namespace xt::data::datasets {
   class SintelStereo : BaseDataset {
/*
       """Sintel `Stereo Dataset <http://sintel.is.tue.mpg.de/stereo>`_.

           The dataset is expected to have the following structure: ::

               root
                   Sintel
                       training
                           final_left
                               scene1
                                   img1.png
                                   img2.png
                                   ...
                               ...
                           final_right
                               scene2
                                   img1.png
                                   img2.png
                                   ...
                               ...
                           disparities
                               scene1
                                   img1.png
                                   img2.png
                                   ...
                               ...
                           occlusions
                               scene1
                                   img1.png
                                   img2.png
                                   ...
                               ...
                           outofframe
                               scene1
                                   img1.png
                                   img2.png
                                   ...
                               ...

           Args:
               root (str or ``pathlib.Path``): Root directory where Sintel Stereo is located.
               pass_name (string): The name of the pass to use, either "final", "clean" or "both".
               transforms (callable, optional): A function/transform that takes in a sample and returns a transformed version.
           """

 */
   public :
       SintelStereo(const std::string &root);
       SintelStereo(const std::string &root, DataMode mode);
       SintelStereo(const std::string &root, DataMode mode , bool download);
       SintelStereo(const std::string &root, DataMode mode , bool download, TransformType transforms);

   private :
       void load_data();

       void check_resources();
    };
}
