#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <vector>
#include <string>
#include <iostream>
using namespace std;

class ImageFolderDataset : public torch::data::Dataset<ImageFolderDataset> {
public:
    ImageFolderDataset(const std::string& root) {
        for (const auto& entry : std::filesystem::directory_iterator(root)) {
            if (entry.is_directory()) {
                int label = label_map.size();
                label_map[entry.path().filename().string()] = label;
                for (const auto& img_entry : std::filesystem::directory_iterator(entry.path())) {
                    images.push_back(img_entry.path());
                    labels.push_back(label);
                }
            }
        }
    }

    // Override get() to return image and label
    torch::data::Example<> get(size_t index) override {
//        cout << images[index].string() << endl;
        cv::Mat img = cv::imread(images[index].string());
//        cv::resize(img, img, cv::Size(224, 224)); // Resize if necessary
        auto tensor_image = torch::from_blob(img.data, {1, img.rows, img.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({0, 3, 1, 2}); // Change to CxHxW
        tensor_image = tensor_image.to(torch::kFloat) / 255.0; // Normalize to [0, 1]

        return {tensor_image.clone(), torch::tensor(labels[index])}; // Clone to keep data alive
    }

    // Override size() to return the number of samples
    std::optional<size_t> size() const override {
        return images.size();
    }

private:
    std::vector<std::filesystem::path> images;
    std::vector<int> labels;
    std::unordered_map<std::string, int> label_map;
};

int main() {
    // Example usage
    ImageFolderDataset dataset("/home/kami/datasets/tiny-imagenet-200/train/");
    auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset.map(torch::data::transforms::Stack<>())), /*batch_size=*/64);

    for (auto& batch : *data_loader) {
        auto images = batch.data;
        auto labels = batch.target;
        cout << images << endl;
//        break;
//        cout << labels << endl;
        // Process your images and labels
    }

    return 0;
}
