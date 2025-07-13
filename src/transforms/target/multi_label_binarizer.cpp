#include "include/transforms/target/multi_label_binarizer.h"

#include <stdexcept>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

int main() {
    // 1. --- Setup ---
    // First, define the complete vocabulary for our movie genre problem.
    // This map must contain all possible genres.
    std::unordered_map<std::string, long> genre_vocab = {
        {"Action",    0},
        {"Adventure", 1},
        {"Comedy",    2},
        {"Drama",     3},
        {"Sci-Fi",    4},
        {"Thriller",  5}
    };
    int num_total_genres = genre_vocab.size();

    xt::transforms::target::MultiLabelBinarizer binarizer(genre_vocab);
    std::cout << "Binarizer created for " << num_total_genres << " total genres." << std::endl;

    // 2. --- Define a sample's active labels ---
    // This movie is an Action, Adventure, and Sci-Fi film.
    std::vector<std::string> active_labels = {"Action", "Adventure", "Sci-Fi"};
    std::cout << "\nInput active labels: {Action, Adventure, Sci-Fi}" << std::endl;

    // 3. --- Run the Transform ---
    auto binarized_any = binarizer.forward({active_labels});

    // 4. --- Verify the Output ---
    try {
        auto binarized_tensor = std::any_cast<torch::Tensor>(binarized_any);

        std::cout << "Output multi-hot tensor: " << binarized_tensor << std::endl;

        // Expected output: [1., 1., 0., 0., 1., 0.]
        // Indices 0 (Action), 1 (Adventure), and 4 (Sci-Fi) should be 1.
        if (binarized_tensor.size(0) == num_total_genres &&
            binarized_tensor[0].item<float>() == 1.0f &&
            binarized_tensor[1].item<float>() == 1.0f &&
            binarized_tensor[4].item<float>() == 1.0f &&
            binarized_tensor[2].item<float>() == 0.0f) {

            std::cout << "Verification successful!" << std::endl;
        }
    } catch (const std::bad_any_cast& e) {
        std::cerr << "Failed to cast result to torch::Tensor." << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::target {

    MultiLabelBinarizer::MultiLabelBinarizer(const std::unordered_map<std::string, long>& class_to_index)
        : class_to_index_(class_to_index), num_classes_(class_to_index.size()) {

        if (num_classes_ == 0) {
            throw std::invalid_argument("class_to_index map cannot be empty.");
        }
    }

    auto MultiLabelBinarizer::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("MultiLabelBinarizer::forward received an empty list.");
        }

        std::vector<std::string> active_labels;
        try {
            active_labels = std::any_cast<std::vector<std::string>>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to MultiLabelBinarizer must be a std::vector<std::string>.");
        }

        // 2. --- Core Logic ---
        // Create a 1D tensor of zeros with length equal to the total number of classes.
        torch::Tensor multi_hot_tensor = torch::zeros({num_classes_}, torch::kFloat32);

        // Iterate through the active labels for the current sample.
        for (const auto& label : active_labels) {
            auto it = class_to_index_.find(label);
            if (it != class_to_index_.end()) {
                // If the label is in our vocabulary, set the corresponding index to 1.0.
                long index = it->second;
                multi_hot_tensor[index] = 1.0f;
            }
            // Note: By convention, labels not in the vocabulary are simply ignored.
        }

        return multi_hot_tensor;
    }

} // namespace xt::transforms::target