#include <transforms/text/vocab_transform.h>

#include <stdexcept>
#include <fstream>

// Example Main - Uncomment to run a standalone test
/*
#include <iostream>

// Mock xt::Module for testing purposes
class xt::Module {
public:
    virtual ~Module() = default;
    virtual auto forward(std::initializer_list<std::any> tensors) -> std::any = 0;
};

// Create a dummy vocab file for the example
void create_dummy_vocab_file() {
    std::ofstream vocab_file("my_vocab.txt");
    vocab_file << "[PAD]\n";      // 0
    vocab_file << "[UNK]\n";      // 1
    vocab_file << "[CLS]\n";      // 2
    vocab_file << "[SEP]\n";      // 3
    vocab_file << "hello\n";      // 4
    vocab_file << "world\n";      // 5
    vocab_file.close();
}

int main() {
    // 1. --- Setup: Create a dummy vocab file ---
    create_dummy_vocab_file();
    std::cout << "Created 'my_vocab.txt' for testing." << std::endl;

    // 2. --- Instantiate the transform with the path ---
    xt::transforms::text::VocabTransform vocab_loader("my_vocab.txt");

    // 3. --- Run the transform to load the file ---
    // Note: No input tensor is needed, so we pass an empty list {}.
    std::cout << "\nLoading vocabulary from file..." << std::endl;
    auto vocab_any = vocab_loader.forward({});

    // 4. --- Cast and verify the result ---
    try {
        auto vocab_map = std::any_cast<std::unordered_map<std::string, long>>(vocab_any);

        std::cout << "Successfully loaded vocabulary with " << vocab_map.size() << " entries." << std::endl;

        // Check a few values to confirm correctness
        std::cout << "ID for '[UNK]': " << vocab_map.at("[UNK]") << std::endl;
        std::cout << "ID for 'hello': " << vocab_map.at("hello") << std::endl;

        if (vocab_map.at("[UNK]") == 1 && vocab_map.at("hello") == 4) {
             std::cout << "\nVerification successful!" << std::endl;
        }

    } catch (const std::bad_any_cast& e) {
        std::cerr << "Failed to cast the result to the expected map type." << std::endl;
        return 1;
    }

    return 0;
}
*/

namespace xt::transforms::text {

    VocabTransform::VocabTransform(const std::string& vocab_path)
            : vocab_path_(vocab_path) {
        if (vocab_path_.empty()) {
            throw std::invalid_argument("Vocabulary file path cannot be empty.");
        }
    }

    auto VocabTransform::forward(std::initializer_list<std::any> tensors) -> std::any {
        // This transform acts as a "source" and doesn't use the input `tensors`.

        std::unordered_map<std::string, long> vocab_map;

        std::ifstream vocab_file(vocab_path_);
        if (!vocab_file.is_open()) {
            throw std::runtime_error("Vocabulary file could not be opened at: " + vocab_path_);
        }

        std::string token;
        long index = 0;

        // Read the file line by line
        while (std::getline(vocab_file, token)) {
            // Remove trailing newline/carriage return characters, which can
            // appear in files created on different operating systems.
            if (!token.empty() && token.back() == '\r') {
                token.pop_back();
            }

            // The line content is the token, and the line number is the ID.
            if (!token.empty()) {
                vocab_map[token] = index++;
            }
        }

        return vocab_map;
    }

} // namespace xt::transforms::text