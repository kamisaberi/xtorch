#pragma once
#include "base.h"
#include "../headers/datasets.h"


namespace xt::data::datasets {
    class HD1K : BaseDataset {
        /*
        Hello,

        Thanks for your interest in our dataset.
        You may download our packages using the following links:

        Complete package with all available data (ca. 8 GB):
        http://hci-benchmark.iwr.uni-heidelberg.de/media/downloads/hd1k_full_package.zip

        Individual packages:

        Benchmark package with input images of the test frames:
        http://hci-benchmark.iwr.uni-heidelberg.de/media/downloads/hd1k_challenge.zip

        Input images for optical flow:
        http://hci-benchmark.iwr.uni-heidelberg.de/media/downloads/hd1k_input.zip

        Ground truth for optical flow:
        http://hci-benchmark.iwr.uni-heidelberg.de/media/downloads/hd1k_flow_gt.zip

        Ground truth uncertainty maps for optical flow:
        http://hci-benchmark.iwr.uni-heidelberg.de/media/downloads/hd1k_flow_uncertainty.zip


        For more information, visit www.hci-benchmark.org or contact us via email.
        We are happy to hear about your feedback, wishes, and bug reports.

        Yours,
        The HD1K Benchmark Team

        */
    public :
        HD1K(const std::string &root);
        HD1K(const std::string &root, DataMode mode);
        HD1K(const std::string &root, DataMode mode , bool download);
        HD1K(const std::string &root, DataMode mode , bool download, vector<std::function<torch::Tensor(torch::Tensor)>> transforms);


    private :
        void load_data();

        void check_resources();
    };
}
