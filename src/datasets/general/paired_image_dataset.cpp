

#include "include/datasets/general/paired_image_dataset.h"

namespace xt::datasets {


        PairedImageDataset::PairedImageDataset (
        const std::string &input_dir,
        const std::string& target_dir
        )
            {
                for (const auto& entry : std::filesystem::directory_iterator(input_dir))
                {
                    if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")
                    {
                        input_paths_.push_back(entry.path().string());
                        std::string target_path = target_dir + "/" + entry.path().filename().string();
                        target_paths_.push_back(target_path);
                    }
                }
            }

        torch::data::Example<> PairedImageDataset::get(size_t index) override
        {
            // Load input image
            cv::Mat input_img = cv::imread(input_paths_[index % input_paths_.size()], cv::IMREAD_GRAYSCALE);
            if (input_img.empty())
            {
                throw std::runtime_error(
                        "Failed to load input image: " + input_paths_[index % input_paths_.size()]);
            }
            input_img.convertTo(input_img, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
            torch::Tensor input_tensor = torch::from_blob(input_img.data, {1, input_img.rows, input_img.cols},
                                                          torch::kFloat32);

            // Load target image
            cv::Mat target_img = cv::imread(target_paths_[index % target_paths_.size()], cv::IMREAD_GRAYSCALE);
            if (target_img.empty())
            {
                throw std::runtime_error(
                        "Failed to load target image: " + target_paths_[index % target_paths_.size()]);
            }
            target_img.convertTo(target_img, CV_32F, 2.0 / 255.0, -1.0); // Normalize to [-1, 1]
            torch::Tensor target_tensor = torch::from_blob(target_img.data, {1, target_img.rows, target_img.cols},
                                                           torch::kFloat32);

            return {input_tensor, target_tensor};
        }

        torch::optional<size_t> PairedImageDataset::size() const override
        {
            return input_paths_.size();
        }

}



