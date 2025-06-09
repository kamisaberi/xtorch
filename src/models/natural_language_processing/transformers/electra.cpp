#include "include/models/natural_language_processing/transformers/electra.h"


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

// Vocabulary class for token-to-ID mapping
class Vocabulary {
public:
    std::map<std::string, int> word_to_id;
    std::map<int, std::string> id_to_word;
    int vocab_size = 0;
    const int unk_id = 0; // [UNK]
    const int cls_id = 1; // [CLS]
    const int sep_id = 2; // [SEP]
    const int mask_id = 3; // [MASK]

    Vocabulary() {
        word_to_id["[UNK]"] = unk_id;
        id_to_word[unk_id] = "[UNK]";
        word_to_id["[CLS]"] = cls_id;
        id_to_word[cls_id] = "[CLS]";
        word_to_id["[SEP]"] = sep_id;
        id_to_word[sep_id] = "[SEP]";
        word_to_id["[MASK]"] = mask_id;
        id_to_word[mask_id] = "[MASK]";
        vocab_size = 4;
    }

    void build(const std::vector<std::string>& sentences, int min_count = 2) {
        std::map<std::string, int> word_counts;
        for (const auto& sentence : sentences) {
            std::istringstream iss(sentence);
            std::string word;
            while (iss >> word) {
                word_counts[word]++;
            }
        }
        for (const auto& pair : word_counts) {
            if (pair.second >= min_count) {
                word_to_id[pair.first] = vocab_size;
                id_to_word[vocab_size] = pair.first;
                vocab_size++;
            }
        }
    }

    int get_id(const std::string& word) const {
        auto it = word_to_id.find(word);
        return it != word_to_id.end() ? it->second : unk_id;
    }
};

// Transformer Embedding Layer
struct TransformerEmbedding : torch::nn::Module {
    torch::nn::Embedding word_embeddings{nullptr};
    torch::nn::Embedding position_embeddings{nullptr};
    torch::nn::LayerNorm layer_norm{nullptr};
    torch::nn::Dropout dropout{nullptr};

    TransformerEmbedding(int vocab_size, int hidden_size, int max_position, float dropout_p = 0.1) {
        word_embeddings = register_module("word_embeddings", torch::nn::Embedding(vocab_size, hidden_size));
        position_embeddings = register_module("position_embeddings", torch::nn::Embedding(max_position, hidden_size));
        layer_norm = register_module("layer_norm", torch::nn::LayerNorm(hidden_size));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor input_ids) {
        auto batch_size = input_ids.size(0);
        auto seq_len = input_ids.size(1);
        auto embeddings = word_embeddings->forward(input_ids);
        auto positions = torch::arange(seq_len, torch::kInt64).unsqueeze(0).repeat({batch_size, 1}).to(input_ids.device());
        embeddings = embeddings + position_embeddings->forward(positions);
        embeddings = layer_norm->forward(embeddings);
        embeddings = dropout->forward(embeddings);
        return embeddings;
    }
};

// Multi-Head Self-Attention
struct MultiHeadAttention : torch::nn::Module {
    int num_heads;
    int head_size;
    torch::nn::Linear query{nullptr}, key{nullptr}, value{nullptr}, out{nullptr};
    torch::nn::Dropout dropout{nullptr};

    MultiHeadAttention(int hidden_size, int num_heads, float dropout_p = 0.1)
        : num_heads(num_heads), head_size(hidden_size / num_heads) {
        query = register_module("query", torch::nn::Linear(hidden_size, hidden_size));
        key = register_module("key", torch::nn::Linear(hidden_size, hidden_size));
        value = register_module("value", torch::nn::Linear(hidden_size, hidden_size));
        out = register_module("out", torch::nn::Linear(hidden_size, hidden_size));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {}) {
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);

        auto q = query->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
        auto k = key->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);
        auto v = value->forward(x).view({batch_size, seq_len, num_heads, head_size}).transpose(1, 2);

        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<float>(head_size));
        if (mask.defined()) {
            scores = scores + (mask * -1e9);
        }
        scores = torch::softmax(scores, -1);
        scores = dropout->forward(scores);
        auto context = torch::matmul(scores, v).transpose(1, 2).contiguous().view({batch_size, seq_len, -1});
        return out->forward(context);
    }
};

// Feed-Forward Network
struct FeedForward : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    torch::nn::Dropout dropout{nullptr};

    FeedForward(int hidden_size, int intermediate_size, float dropout_p = 0.1) {
        fc1 = register_module("fc1", torch::nn::Linear(hidden_size, intermediate_size));
        fc2 = register_module("fc2", torch::nn::Linear(intermediate_size, hidden_size));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::gelu(fc1->forward(x));
        x = dropout->forward(x);
        x = fc2->forward(x);
        return x;
    }
};

// Transformer Layer
struct TransformerLayer : torch::nn::Module {
    MultiHeadAttention attention{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr};
    FeedForward ffn{nullptr};
    torch::nn::Dropout dropout{nullptr};

    TransformerLayer(int hidden_size, int num_heads, int intermediate_size, float dropout_p = 0.1)
        : attention(MultiHeadAttention(hidden_size, num_heads, dropout_p)),
          ffn(FeedForward(hidden_size, intermediate_size, dropout_p)) {
        norm1 = register_module("norm1", torch::nn::LayerNorm(hidden_size));
        norm2 = register_module("norm2", torch::nn::LayerNorm(hidden_size));
        dropout = register_module("dropout", torch::nn::Dropout(dropout_p));
    }

    torch::Tensor forward(torch::Tensor x, torch::Tensor mask = {}) {
        auto attn_output = attention->forward(x, mask);
        x = norm1->forward(x + dropout->forward(attn_output));
        auto ffn_output = ffn->forward(x);
        x = norm2->forward(x + dropout->forward(ffn_output));
        return x;
    }
};

// Generator Model (small transformer for MLM)
struct Generator : torch::nn::Module {
    TransformerEmbedding embedding{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::Linear mlm_head{nullptr};
    torch::nn::LayerNorm mlm_norm{nullptr};

    Generator(int vocab_size, int hidden_size, int num_heads, int intermediate_size, int num_layers, int max_position) {
        embedding = register_module("embedding", TransformerEmbedding(vocab_size, hidden_size, max_position));
        layers = register_module("layers", torch::nn::ModuleList());
        for (int i = 0; i < num_layers; ++i) {
            layers->push_back(TransformerLayer(hidden_size, num_heads, intermediate_size));
        }
        mlm_head = register_module("mlm_head", torch::nn::Linear(hidden_size, vocab_size));
        mlm_norm = register_module("mlm_norm", torch::nn::LayerNorm(hidden_size));
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor attention_mask = {}) {
        auto x = embedding->forward(input_ids);
        for (auto& layer : *layers) {
            x = layer->as<TransformerLayer>()->forward(x, attention_mask);
        }
        x = mlm_norm->forward(x);
        return mlm_head->forward(x);
    }
};

// Discriminator Model (larger transformer for RTD)
struct Discriminator : torch::nn::Module {
    TransformerEmbedding embedding{nullptr};
    torch::nn::ModuleList layers{nullptr};
    torch::nn::Linear rtd_head{nullptr};

    Discriminator(int vocab_size, int hidden_size, int num_heads, int intermediate_size, int num_layers, int max_position) {
        embedding = register_module("embedding", TransformerEmbedding(vocab_size, hidden_size, max_position));
        layers = register_module("layers", torch::nn::ModuleList());
        for (int i = 0; i < num_layers; ++i) {
            layers->push_back(TransformerLayer(hidden_size, num_heads, intermediate_size));
        }
        rtd_head = register_module("rtd_head", torch::nn::Linear(hidden_size, 1));
    }

    torch::Tensor forward(torch::Tensor input_ids, torch::Tensor attention_mask = {}) {
        auto x = embedding->forward(input_ids);
        for (auto& layer : *layers) {
            x = layer->as<TransformerLayer>()->forward(x, attention_mask);
        }
        return torch::sigmoid(rtd_head->forward(x)).squeeze(-1);
    }
};

// ELECTRA Model (combines generator and discriminator)
struct ELECTRA : torch::nn::Module {
    std::shared_ptr<Generator> generator{nullptr};
    std::shared_ptr<Discriminator> discriminator{nullptr};

    ELECTRA(int vocab_size, int gen_hidden_size, int gen_num_heads, int gen_intermediate_size, int gen_num_layers,
            int disc_hidden_size, int disc_num_heads, int disc_intermediate_size, int disc_num_layers, int max_position) {
        generator = register_module("generator", std::make_shared<Generator>(
            vocab_size, gen_hidden_size, gen_num_heads, gen_intermediate_size, gen_num_layers, max_position));
        discriminator = register_module("discriminator", std::make_shared<Discriminator>(
            vocab_size, disc_hidden_size, disc_num_heads, disc_intermediate_size, disc_num_layers, max_position));
    }
};

// Dataset for ELECTRA RTD
struct ELECTRADataset {
    std::vector<std::vector<int>> sentences;
    Vocabulary vocab;
    const int max_len = 128;
    const float mask_prob = 0.15;

    void load_data(const std::string& filename) {
        std::ifstream file(filename);
        std::string line;
        std::vector<std::string> raw_sentences;
        while (getline(file, line)) {
            raw_sentences.push_back(line);
        }
        vocab.build(raw_sentences);
        for (const auto& sentence : raw_sentences) {
            std::istringstream iss(sentence);
            std::string word;
            std::vector<int> tokens = {vocab.cls_id};
            while (iss >> word && tokens.size() < max_len - 1) {
                tokens.push_back(vocab.get_id(word));
            }
            tokens.push_back(vocab.sep_id);
            sentences.push_back(tokens);
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size, Generator& generator, torch::Device device) {
        std::vector<torch::Tensor> inputs, labels, attention_masks;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0, 1.0);

        size_t max_batch_len = 0;
        for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
            max_batch_len = std::max(max_batch_len, sentences[i].size());
        }

        for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
            auto tokens = sentences[i];
            std::vector<int> input_tokens = tokens;
            std::vector<int> disc_labels(max_batch_len, 0); // 0: real, 1: fake
            std::vector<int> attention_mask(max_batch_len, 1);

            // Mask tokens for generator
            std::vector<int> masked_tokens = tokens;
            std::vector<bool> is_masked(tokens.size(), false);
            for (size_t j = 1; j < tokens.size() - 1; ++j) { // Skip [CLS] and [SEP]
                if (dist(gen) < mask_prob) {
                    masked_tokens[j] = vocab.mask_id;
                    is_masked[j] = true;
                }
            }

            // Generate fake tokens using generator
            auto gen_input = torch::tensor(masked_tokens, torch::kInt64).unsqueeze(0).to(device);
            auto gen_mask = torch::tensor(attention_mask, torch::kFloat32).unsqueeze(0).unsqueeze(1).unsqueeze(2).to(device);
            torch::NoGradGuard no_grad;
            auto gen_logits = generator.forward(gen_input, gen_mask);
            auto gen_probs = torch::softmax(gen_logits, -1);
            auto sampled_ids = torch::multinomial(gen_probs.squeeze(0), 1).squeeze(-1).cpu();
            for (size_t j = 0; j < tokens.size(); ++j) {
                if (is_masked[j]) {
                    input_tokens[j] = sampled_ids[j].item<int>();
                    disc_labels[j] = 1; // Fake token
                }
            }

            while (input_tokens.size() < max_batch_len) {
                input_tokens.push_back(0);
                disc_labels.push_back(-100); // Ignore in loss
                attention_mask.push_back(0);
            }

            inputs.push_back(torch::tensor(input_tokens, torch::kInt64));
            labels.push_back(torch::tensor(disc_labels, torch::kFloat32));
            attention_masks.push_back(torch::tensor(attention_mask, torch::kFloat32));
        }

        return {torch::stack(inputs), torch::stack(labels), torch::stack(attention_masks)};
    }
};

int main() {
    torch::manual_seed(0);
    srand(static_cast<unsigned>(time(0)));

    // Load dataset
    ELECTRADataset dataset;
    dataset.load_data("corpus.txt"); // One sentence per line

    // Initialize model
    ELECTRA model(
        dataset.vocab.vocab_size, 128, 2, 512, 2, // Generator: smaller
        256, 4, 1024, 6, // Discriminator: larger
        128
    );
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model.to(device);

    // Optimizers
    torch::optim::Adam gen_optimizer(model->generator->parameters(), torch::optim::AdamOptions(0.0001).weight_decay(0.01));
    torch::optim::Adam disc_optimizer(model->discriminator->parameters(), torch::optim::AdamOptions(0.0001).weight_decay(0.01));

    // Training loop
    const int epochs = 10;
    const size_t batch_size = 16;
    const float disc_weight = 50.0; // Weight for discriminator loss
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_gen_loss = 0.0, total_disc_loss = 0.0;
        size_t num_batches = 0;
        for (size_t i = 0; i < dataset.sentences.size(); i += batch_size) {
            auto [input_ids, disc_labels, attention_mask] = dataset.get_batch(i, batch_size, *model->generator, device);
            input_ids = input_ids.to(device);
            disc_labels = disc_labels.to(device);
            attention_mask = attention_mask.to(device).unsqueeze(1).unsqueeze(2);

            // Train discriminator
            disc_optimizer.zero_grad();
            auto disc_logits = model->discriminator->forward(input_ids, attention_mask);
            auto disc_loss = torch::binary_cross_entropy(disc_logits, disc_labels);
            disc_loss.backward();
            disc_optimizer.step();

            // Train generator
            gen_optimizer.zero_grad();
            auto gen_logits = model->generator->forward(input_ids, attention_mask);
            auto disc_pred = model->discriminator->forward(input_ids, attention_mask).detach();
            auto gen_loss = torch::nll_loss(torch::log_softmax(gen_logits.view(-1, dataset.vocab.vocab_size), -1), disc_labels.view(-1).to(torch::kInt64));
            gen_loss.backward();
            gen_optimizer.step();

            total_gen_loss += gen_loss.item<float>();
            total_disc_loss += disc_loss.item<float>();
            num_batches++;
        }
        std::cout << "Epoch: " << epoch + 1
                  << ", Gen Loss: " << total_gen_loss / num_batches
                  << ", Disc Loss: " << total_disc_loss / num_batches << std::endl;
    }

    // Save model
    torch::save(model, "electra_model.pt");

    return 0;
}


namespace xt::models
{
    ELECTRA::ELECTRA(int num_classes, int in_channels)
    {
    }

    ELECTRA::ELECTRA(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void ELECTRA::reset()
    {
    }

    auto ELECTRA::forward(std::initializer_list<std::any> tensors) -> std::any
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
