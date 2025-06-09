#include "include/models/natural_language_processing/others/word2vec.h"


using namespace std;


#include <torch/torch.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <map>
#include <algorithm>

// Vocabulary class for word-to-ID mapping
class Vocabulary {
public:
    std::map<std::string, int> word_to_id;
    std::map<int, std::string> id_to_word;
    std::vector<float> word_freqs;
    int vocab_size = 0;
    const int unk_id = 0; // <unk>

    Vocabulary() {
        word_to_id["<unk>"] = unk_id;
        id_to_word[unk_id] = "<unk>";
        vocab_size = 1;
    }

    void build(const std::vector<std::string>& sentences, int min_count = 5) {
        std::map<std::string, int> word_counts;
        float total_count = 0.0;
        for (const auto& sentence : sentences) {
            std::istringstream iss(sentence);
            std::string word;
            while (iss >> word) {
                word_counts[word]++;
                total_count += 1.0;
            }
        }
        for (const auto& pair : word_counts) {
            if (pair.second >= min_count) {
                word_to_id[pair.first] = vocab_size;
                id_to_word[vocab_size] = pair.first;
                word_freqs.push_back(std::pow(pair.second / total_count, 0.75)); // Subsampling freq
                vocab_size++;
            }
        }
        // Normalize frequencies
        float sum_freqs = std::accumulate(word_freqs.begin(), word_freqs.end(), 0.0);
        for (auto& freq : word_freqs) {
            freq /= sum_freqs;
        }
    }

    int get_id(const std::string& word) const {
        auto it = word_to_id.find(word);
        return it != word_to_id.end() ? it->second : unk_id;
    }
};

// Skip-Gram Dataset
struct SkipGramDataset {
    std::vector<std::pair<int, int>> pairs; // (target_word_id, context_word_id)
    Vocabulary vocab;
    const int window_size = 5;
    const int num_negative = 5;

    void build(const std::string& filename) {
        // Load sentences
        std::ifstream file(filename);
        std::string line;
        std::vector<std::string> sentences;
        while (std::getline(file, line)) {
            if (!line.empty()) {
                sentences.push_back(line);
            }
        }
        vocab.build(sentences);

        // Generate target-context pairs
        for (const auto& sentence : sentences) {
            std::istringstream iss(sentence);
            std::string word;
            std::vector<int> word_ids;
            while (iss >> word) {
                int word_id = vocab.get_id(word);
                if (word_id != vocab.unk_id) {
                    word_ids.push_back(word_id);
                }
            }
            for (size_t i = 0; i < word_ids.size(); ++i) {
                int target_id = word_ids[i];
                for (int j = std::max(0, static_cast<int>(i) - window_size);
                     j <= std::min(static_cast<int>(i) + window_size, static_cast<int>(word_ids.size()) - 1); ++j) {
                    if (i == j) continue;
                    int context_id = word_ids[j];
                    pairs.emplace_back(target_id, context_id);
                }
            }
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size,
                                                                     std::mt19937& gen) {
        std::vector<int> target_ids, context_ids, negative_ids;
        std::discrete_distribution<int> dist(vocab.word_freqs.begin(), vocab.word_freqs.end());

        for (size_t i = idx; i < std::min(idx + batch_size, pairs.size()); ++i) {
            auto [target_id, context_id] = pairs[i];
            target_ids.push_back(target_id);
            context_ids.push_back(context_id);
            // Sample negative examples
            for (int k = 0; k < num_negative; ++k) {
                int neg_id;
                do {
                    neg_id = dist(gen);
                } while (neg_id == context_id || neg_id == vocab.unk_id);
                negative_ids.push_back(neg_id);
            }
        }

        return {
            torch::tensor(target_ids, torch::kInt64),
            torch::tensor(context_ids, torch::kInt64),
            torch::tensor(negative_ids, torch::kInt64).view({-1, num_negative})
        };
    }
};

// Word2Vec Skip-Gram Model
struct Word2Vec : torch::nn::Module {
    torch::nn::Embedding target_embeddings{nullptr};
    torch::nn::Embedding context_embeddings{nullptr};

    Word2Vec(int vocab_size, int embedding_size) {
        target_embeddings = register_module("target_embeddings",
                                           torch::nn::Embedding(vocab_size, embedding_size));
        context_embeddings = register_module("context_embeddings",
                                            torch::nn::Embedding(vocab_size, embedding_size));

        // Initialize embeddings
        torch::nn::init::uniform_(target_embeddings->weight, -0.5 / embedding_size, 0.5 / embedding_size);
        torch::nn::init::uniform_(context_embeddings->weight, -0.5 / embedding_size, 0.5 / embedding_size);
    }

    std::tuple<torch::Tensor, torch::Tensor> forward(torch::Tensor target_ids, torch::Tensor context_ids,
                                                    torch::Tensor negative_ids) {
        auto target_emb = target_embeddings->forward(target_ids); // [batch, embedding_size]
        auto context_emb = context_embeddings->forward(context_ids); // [batch, embedding_size]
        auto negative_emb = context_embeddings->forward(negative_ids); // [batch, num_negative, embedding_size]

        // Positive score: dot product
        auto positive_score = (target_emb * context_emb).sum(-1); // [batch]
        positive_score = torch::log_sigmoid(positive_score); // [batch]

        // Negative score
        auto negative_score = (target_emb.unsqueeze(1) * negative_emb).sum(-1); // [batch, num_negative]
        negative_score = torch::log_sigmoid(-negative_score).sum(-1); // [batch]

        return {positive_score, negative_score};
    }
};

// Word2Vec Loss Function
struct Word2VecLoss : torch::nn::Module {
    torch::Tensor forward(torch::Tensor positive_score, torch::Tensor negative_score) {
        return -(positive_score + negative_score).mean();
    }
};

int main() {
    torch::manual_seed(0);
    std::random_device rd;
    std::mt19937 gen(rd());

    // Load dataset
    SkipGramDataset dataset;
    dataset.build("corpus.txt"); // One sentence per line

    // Initialize model
    Word2Vec model(dataset.vocab.vocab_size, 100); // 100-dimensional embeddings
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model.to(device);

    // Optimizer and loss
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));
    Word2VecLoss loss_fn;

    // Training loop
    const int epochs = 20;
    const size_t batch_size = 512;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0;
        size_t num_batches = 0;
        model.train();

        for (size_t i = 0; i < dataset.pairs.size(); i += batch_size) {
            auto [target_ids, context_ids, negative_ids] = dataset.get_batch(i, batch_size, gen);
            target_ids = target_ids.to(device);
            context_ids = context_ids.to(device);
            negative_ids = negative_ids.to(device);

            optimizer.zero_grad();
            auto [positive_score, negative_score] = model.forward(target_ids, context_ids, negative_ids);
            auto loss = loss_fn.forward(positive_score, negative_score);
            loss.backward();
            optimizer.step();

            total_loss += loss.item<float>();
            num_batches++;
        }

        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
    }

    // Save embeddings
    torch::save(model.target_embeddings->weight, "word2vec_embeddings.pt");

    return 0;
}

namespace xt::models
{
    Word2Vec::Word2Vec(int num_classes, int in_channels)
    {
    }

    Word2Vec::Word2Vec(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void Word2Vec::reset()
    {
    }

    auto Word2Vec::forward(std::initializer_list<std::any> tensors) -> std::any
    {
        std::vector<std::any> any_vec(tensors);

        std::vector<torch::Tensor> tensor_vec;
        for (const auto& item : any_vec)
        {
            tensor_vec.push_back(std::any_cast<torch::Tensor>(item));
        }

        torch::Tensor x = tensor_vec[0];

        return x;
    }
}
