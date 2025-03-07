#pragma once

#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class USPS : torch::data::Dataset<USPS> {
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

//       split_list = {
//               "train": [
//               "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2",
//               "usps.bz2",
//               "ec16c51db3855ca6c91edd34d0e9b197",
//               ],
//               "test": [
//               "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2",
//               "usps.t.bz2",
//               "8ea070ee2aca1ac39742fdd1ef5ed118",
//               ],
//       }


    public :
        USPS();
    };
}
