#pragma once

//#include <vector>
#include <fstream>
//#include <iostream>
//#include <string>
#include <filesystem>
//#include <curl/curl.h>
#include <torch/torch.h>
#include <vector>
#include <tuple>
#include <map>
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
#include "base.h"
#include "../utils/downloader.h"
#include "../utils/archiver.h"
#include "../utils/md5.h"
#include "../exceptions/implementation.h"
#include "../definitions/transforms.h"
#include "../types/arguments.h"


namespace torch::ext::data {
    torch::data::datasets::MapDataset<torch::ext::data::datasets::BaseDataset , torch::data::transforms::Stack<>()>
    transform_dataset(torch::ext::data::datasets::BaseDataset, vector<torch::data::transforms::Lambda<torch::data::Example<> > > transforms);
}
