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
//#include <fstream>
//#include <iostream>
//#include <string>
//#include <filesystem>
#include "../utils/downloader.h"
#include "../utils/archiver.h"
#include "../utils/md5.h"
#include "../exceptions/implementation.h"

using namespace std;
namespace fs = std::filesystem;


namespace torch::ext::data::datasets {
    class MNIST : torch::data::Dataset<MNIST> {

    public :
        MNIST(const std::string &root , bool train= true, bool download = false);

        torch::data::Example<> get(size_t index) override;

        torch::optional<size_t> size() const override;
    private :
        std::vector<torch::Tensor> data; // Store image data as tensors
        std::vector<int64_t> labels;      // Store labels
        std::string download_url_base = "https://ossci-datasets.s3.amazonaws.com/mnist/";
        fs::path root;
        fs::path dataset_path;
        fs::path dataset_folder_name = "MNIST/raw";

        vector<tuple<fs::path ,std::string >> resources = {
                {fs::path("train-images-idx3-ubyte.gz"),"f68b3c2dcbeaaa9fbdd348bbdeb94873" },
                {fs::path("train-labels-idx1-ubyte.gz"),"d53e105ee54ea40749a09fcbcd1e9432" },
                {fs::path("t10k-images-idx3-ubyte.gz"),"9fb629c4189551a2d022fa330f9573f3" },
                {fs::path("t10k-labels-idx1-ubyte.gz"),"ec29112dd5afa0611ce80d1b7f02629c" },
        };

        void load_data(bool train = true);


    };
}
