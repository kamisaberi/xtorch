#include <transforms/text/str_to_int_transform.h>


// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

// Helper to print a vector
template<typename T>
void print_vector(const std::string& name, const std::vector<T>& vec) {
    std::cout << name << " (size " << vec.size() << "): [ ";
    for (const auto& val : vec) {
        std::cout << val << " ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    // 1. --- Create a sample vocabulary ---
    std::unordered_map<std::string, long> my_vocab = {
        {"[PAD]", 0},
        {"[UNK]", 1},
        {"[CLS]", 2},
        {"the", 3},
        {"quick", 4},
        {"brown", 5},
        {"fox", 6}
    };
    std::cout << "Vocabulary created with " << my_vocab.size() << " entries." << std::endl;

    // 2. --- Instantiate the transform with the vocabulary ---
    xt::transforms::text::StrToIntTransform to_int_transformer(my_vocab);

    // 3. --- Create a sample list of tokens ---
    // Note: "jumps" is not in the vocabulary.
    std::vector<std::string> tokens = {"[CLS]", "the", "quick", "brown", "fox", "jumps"};
    print_vector("\nInput Tokens", tokens);

    // 4. --- Run the transform ---
    auto ids_any = to_int_transformer.forward({tokens});
    auto ids = std::any_cast<std::vector<long>>(ids_any);

    // 5. --- Print and verify the result ---
    // Expected output: [ 2 3 4 5 6 1 ] (since "jumps" maps to the [UNK] id of 1)
    print_vector("Output IDs", ids);

    if (ids.back() == 1) {
        std::cout << "\nCorrectly mapped out-of-vocabulary token 'jumps' to [UNK] ID (1)." << std::endl;
    }

    return 0;
}
*/

namespace xt::transforms::text {

    StrToIntTransform::StrToIntTransform(const std::unordered_map<std::string, long>& vocab, const std::string& unk_token)
            : vocab_(vocab) {
        // Find and store the ID for the unknown token. This is more efficient
        // than looking up the string "[UNK]" every single time.
        auto it = vocab_.find(unk_token);
        if (it == vocab_.end()) {
            throw std::invalid_argument("The unknown token '" + unk_token + "' was not found in the provided vocabulary.");
        }
        unk_id_ = it->second;
    }

    auto StrToIntTransform::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("StrToIntTransform::forward received an empty list.");
        }

        std::vector<std::string> input_tokens;
        try {
            input_tokens = std::any_cast<std::vector<std::string>>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to StrToIntTransform must be of type std::vector<std::string>.");
        }

        // 2. --- Conversion Loop ---
        std::vector<long> output_ids;
        output_ids.reserve(input_tokens.size()); // Pre-allocate memory for efficiency

        for (const auto& token : input_tokens) {
            auto it = vocab_.find(token);
            if (it != vocab_.end()) {
                // Token is in the vocabulary, add its ID.
                output_ids.push_back(it->second);
            } else {
                // Token is not in the vocabulary, use the pre-fetched [UNK] ID.
                output_ids.push_back(unk_id_);
            }
        }

        // 3. --- Return Result ---
        return output_ids;
    }

} // namespace xt::transforms::text