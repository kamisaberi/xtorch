#include <transforms/text/add_token.h>


// Example Main - Uncomment to run a standalone test
// #include <iostream>
//
// // Mock xt::Module for testing purposes
// class xt::Module {
// public:
//     virtual ~Module() = default;
//     virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
// };
//
// void print_tokens(const std::vector<std::string>& tokens) {
//     std::cout << "[ ";
//     for (const auto& token : tokens) {
//         std::cout << "\"" << token << "\" ";
//     }
//     std::cout << "]" << std::endl;
// }
//
// int main() {
//     // 1. Create a dummy list of tokens.
//     std::vector<std::string> tokens = {"hello", "world", "this", "is", "a", "test"};
//     std::cout << "Original tokens: ";
//     print_tokens(tokens);
//
//     // --- Example 1: Add [CLS] token at the beginning ---
//     std::cout << "\n--- Adding [CLS] at the beginning ---" << std::endl;
//     xt::transforms::text::AddToken add_cls_begin("[CLS]", xt::transforms::text::AddTokenLocation::BEGIN);
//
//     std::vector<std::string> tokens_with_cls = std::any_cast<std::vector<std::string>>(add_cls_begin.forward({tokens}));
//
//     std::cout << "Result: ";
//     print_tokens(tokens_with_cls);
//
//     // --- Example 2: Add [SEP] token at the end ---
//     std::cout << "\n--- Adding [SEP] at the end ---" << std::endl;
//     xt::transforms::text::AddToken add_sep_end("[SEP]", xt::transforms::text::AddTokenLocation::END);
//
//     std::vector<std::string> tokens_with_sep = std::any_cast<std::vector<std::string>>(add_sep_end.forward({tokens}));
//
//     std::cout << "Result: ";
//     print_tokens(tokens_with_sep);
//
//     return 0;
// }

namespace xt::transforms::text {

    AddToken::AddToken(const std::string& token, AddTokenLocation location)
            : token_(token), location_(location) {
        if (token.empty()) {
            throw std::invalid_argument("Token cannot be an empty string.");
        }
    }

    auto AddToken::forward(std::initializer_list<std::any> tensors) -> std::any {
        // 1. --- Input Validation ---
        std::vector<std::any> any_vec(tensors);
        if (any_vec.empty()) {
            throw std::invalid_argument("AddToken::forward received an empty list.");
        }

        // We expect the input to be a vector of strings (tokens).
        std::vector<std::string> input_tokens;
        try {
            input_tokens = std::any_cast<std::vector<std::string>>(any_vec[0]);
        } catch (const std::bad_any_cast& e) {
            throw std::invalid_argument("Input to AddToken must be of type std::vector<std::string>.");
        }

        // 2. --- Add the Token ---
        // Create a copy to avoid modifying the original vector if it's used elsewhere.
        std::vector<std::string> output_tokens = input_tokens;

        if (location_ == AddTokenLocation::BEGIN) {
            // Insert the token at the beginning of the vector.
            output_tokens.insert(output_tokens.begin(), token_);
        } else { // location_ == AddTokenLocation::END
            // Append the token to the end of the vector.
            output_tokens.push_back(token_);
        }

        // 3. --- Return the Result ---
        return output_tokens;
    }

} // namespace xt::transforms::text