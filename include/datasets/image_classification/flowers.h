#pragma once
#include "../base/base.h"
#include "../../headers/datasets.h"


namespace xt::data::datasets {
    class Flowers102 : public BaseDataset {
    public :
        explicit Flowers102(const std::string &root);
        Flowers102(const std::string &root, DataMode mode);
        Flowers102(const std::string &root, DataMode mode , bool download);
        Flowers102(const std::string &root, DataMode mode , bool download, TransformType transforms);

    private :

        // _download_url_prefix = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
        // _file_dict = {  # filename, md5
        // "image": ("102flowers.tgz", "52808999861908f626f3c1f4e79d11fa"),
        // "label": ("imagelabels.mat", "e0620be6f572b9609742df49c70aed4d"),
        // "setid": ("setid.mat", "a5357ecc9cb78c4bef273ce3793fc85c"),
        // }
        // _splits_map = {"train": "trnid", "val": "valid", "test": "tstid"}

        void load_data();

        void check_resources();
    };
}
