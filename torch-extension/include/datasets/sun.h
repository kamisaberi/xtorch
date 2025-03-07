#pragma once
#include "../base/datasets.h"


using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class SUN397 : torch::data::Dataset<SUN397> {
    private :
        fs::path url = fs::path("http://vision.princeton.edu/projects/2010/SUN/SUN397.tar.gz");
        std::string md5 = "8ca2778205c41d23104230ba66911c7a";

    public :
        SUN397();
    };
}
