#include "../../../include/datasets/text_classification/db_pedia.h"

namespace xt::data::datasets {

    DBPedia::DBPedia(const std::string &root): DBPedia::DBPedia(root, DataMode::TRAIN, false) {
    }

    DBPedia::DBPedia(const std::string &root, DataMode mode): DBPedia::DBPedia(root, mode, false) {
    }

    DBPedia::DBPedia(const std::string &root, DataMode mode, bool download) : BaseDataset(root, mode, download) {
        throw std::runtime_error("DBPedia: DBPedia not implemented");
    }


    DBPedia::DBPedia(const std::string &root, DataMode mode, bool download,
                           TransformType transforms) : BaseDataset(root, mode, download, transforms) {
        throw std::runtime_error("DBPedia: DBPedia not implemented");
    }


}
