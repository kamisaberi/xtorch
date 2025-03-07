#pragma once
#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class SVHN : torch::data::Dataset<SVHN> {
    private:
        std::map<std::string, std::tuple<fs::path, fs::path, std::string>> resources = {
                {"train", {
                                  fs::path("http://ufldl.stanford.edu/housenumbers/train_32x32.mat"),
                                  fs::path("train_32x32.mat"),
                                  "e26dedcc434d2e4c54c9b2d4a06d8373"}
                },
                {"test",  {
                                  fs::path("http://ufldl.stanford.edu/housenumbers/test_32x32.mat"),
                                  fs::path("test_32x32.mat"),
                                  "eb5a983be6a315427106f1b164d9cef3"}
                },
                {"extra", {
                                  fs::path("http://ufldl.stanford.edu/housenumbers/extra_32x32.mat"),
                                  fs::path("extra_32x32.mat"),
                                  "a93ce644f1a588dc4d68dda5feec44a7"
                          }
                }
        };

    public :
        SVHN();
    };
}
