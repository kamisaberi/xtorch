#include "../../../include/datasets/sentiment_analysis/imdb.h"

namespace xt::data::datasets {

    IMDB::IMDB(const std::string &root): IMDB::IMDB(root, DataMode::TRAIN, false) {
    }

    IMDB::IMDB(const std::string &root, DataMode mode): IMDB::IMDB(root, mode, false) {
    }

    IMDB::IMDB(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("IMDB: IMDB not implemented");
    }


    IMDB::IMDB(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("IMDB: IMDB not implemented");
    }


}
