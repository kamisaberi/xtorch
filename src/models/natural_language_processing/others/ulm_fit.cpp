#include <models/natural_language_processing/others/ulm_fit.h>


using namespace std;


// #include <torch/torch.h>
// #include <vector>
// #include <string>
// #include <fstream>
// #include <sstream>
// #include <random>
// #include <iostream>
// #include <map>
// #include <algorithm>
//
// // Vocabulary class for word-to-ID mapping
// class Vocabulary {
// public:
//     std::map<std::string, int> word_to_id;
//     std::map<int, std::string> id_to_word;
//     int vocab_size = 0;
//     const int unk_id = 0; // <unk>
//     const int pad_id = 1; // <pad>
//     const int bos_id = 2; // <bos>
//     const int eos_id = 3; // <eos>
//
//     Vocabulary() {
//         word_to_id["<unk>"] = unk_id;
//         id_to_word[unk_id] = "<unk>";
//         word_to_id["<pad>"] = pad_id;
//         id_to_word[pad_id] = "<pad>";
//         word_to_id["<bos>"] = bos_id;
//         id_to_word[bos_id] = "<bos>";
//         word_to_id["<eos>"] = eos_id;
//         id_to_word[eos_id] = "<eos>";
//         vocab_size = 4;
//     }
//
//     void build(const std::vector<std::string>& sentences, int min_count = 2) {
//         std::map<std::string, int> word_counts;
//         for (const auto& sentence : sentences) {
//             std::istringstream iss(sentence);
//             std::string word;
//             while (iss >> word) {
//                 word_counts[word]++;
//             }
//         }
//         for (const auto& pair : word_counts) {
//             if (pair.second >= min_count) {
//                 word_to_id[pair.first] = vocab_size;
//                 id_to_word[vocab_size] = pair.first;
//                 vocab_size++;
//             }
//         }
//     }
//
//     int get_id(const std::string& word) const {
//         auto it = word_to_id.find(word);
//         return it != word_to_id.end() ? it->second : unk_id;
//     }
//
//     std::vector<int> tokenize(const std::string& sentence) {
//         std::vector<int> token_ids = {bos_id};
//         std::istringstream iss(sentence);
//         std::string word;
//         while (iss >> word) {
//             token_ids.push_back(get_id(word));
//         }
//         token_ids.push_back(eos_id);
//         return token_ids;
//     }
// };
//
// // AWD-LSTM Module (ASGD Weight-Dropped LSTM)
// struct AWDLSTM : torch::nn::Module {
//     torch::nn::LSTM lstm{nullptr};
//     float weight_drop = 0.0;
//
//     AWDLSTM(int input_size, int hidden_size, int num_layers, float dropout_p = 0.4, float weight_drop_p = 0.5)
//         : weight_drop(weight_drop_p) {
//         lstm = register_module("lstm", torch::nn::LSTM(torch::nn::LSTMOptions(input_size, hidden_size)
//                                                         .num_layers(num_layers)
//                                                         .dropout(dropout_p)
//                                                         .batch_first(true)));
//     }
//
//     std::tuple<torch::Tensor, std::tuple<torch::Tensor, torch::Tensor>> forward(
//         torch::Tensor input, std::tuple<torch::Tensor, torch::Tensor> hidden) {
//         // Apply weight dropout to LSTM weights (simplified)
//         if (training() && weight_drop > 0) {
//             for (auto& param : lstm->named_parameters()) {
//                 if (param.key().find("weight_hh") != std::string::npos) {
//                     auto mask = torch::bernoulli(torch::ones_like(param.value()) * (1 - weight_drop));
//                     param.value().mul_(mask / (1 - weight_drop));
//                 }
//             }
//         }
//         return lstm->forward(input, hidden);
//     }
// };
//
// // ULMFiT Model
// struct ULMFiT : torch::nn::Module {
//     torch::nn::Embedding embedding{nullptr};
//     AWDLSTM encoder{nullptr};
//     torch::nn::Linear lm_head{nullptr};
//     torch::nn::Linear classifier{nullptr};
//     torch::nn::Dropout emb_dropout{nullptr}, out_dropout{nullptr};
//     bool is_classifier_mode = false;
//     int vocab_size, num_classes;
//
//     ULMFiT(int vocab_size_, int emb_size, int hidden_size, int num_layers, int num_classes_)
//         : vocab_size(vocab_size_), num_classes(num_classes_) {
//         embedding = register_module("embedding", torch::nn::Embedding(vocab_size, emb_size));
//         encoder = register_module("encoder", AWDLSTM(emb_size, hidden_size, num_layers));
//         lm_head = register_module("lm_head", torch::nn::Linear(hidden_size, vocab_size));
//         classifier = register_module("classifier", torch::nn::Linear(hidden_size, num_classes));
//         emb_dropout = register_module("emb_dropout", torch::nn::Dropout(0.1));
//         out_dropout = register_module("out_dropout", torch::nn::Dropout(0.4));
//     }
//
//     torch::Tensor forward(torch::Tensor input, torch::Tensor lengths) {
//         auto batch_size = input.size(0);
//         auto seq_len = input.size(1);
//
//         // Embedding
//         auto emb = embedding->forward(input); // [batch, seq_len, emb_size]
//         emb = emb_dropout->forward(emb);
//
//         // Initialize hidden state
//         auto hidden = std::make_tuple(
//             torch::zeros({encoder->lstm->options.num_layers(), batch_size, encoder->lstm->options.hidden_size()}).to(input.device()),
//             torch::zeros({encoder->lstm->options.num_layers(), batch_size, encoder->lstm->options.hidden_size()}).to(input.device())
//         );
//
//         // Encoder (AWD-LSTM)
//         auto packed = torch::nn::utils::rnn::pack_padded_sequence(emb, lengths, true);
//         auto [output, hidden] = encoder->forward(packed, hidden);
//         auto [unpacked, _] = torch::nn::utils::rnn::pad_packed_sequence(output, true); // [batch, seq_len, hidden_size]
//         unpacked = out_dropout->forward(unpacked);
//
//         if (is_classifier_mode) {
//             // Classifier: use last non-padded output
//             std::vector<torch::Tensor> last_outputs;
//             for (int64_t i = 0; i < batch_size; ++i) {
//                 auto len_i = lengths[i].item<int64_t>() - 1; // Adjust for <bos>
//                 last_outputs.push_back(unpacked.select(0, i).select(0, len_i));
//             }
//             auto final_out = torch::stack(last_outputs); // [batch, hidden_size]
//             return classifier->forward(final_out); // [batch, num_classes]
//         } else {
//             // Language model: predict next word
//             return lm_head->forward(unpacked); // [batch, seq_len, vocab_size]
//         }
//     }
//
//     void set_classifier_mode(bool mode) {
//         is_classifier_mode = mode;
//     }
// };
//
// // Language Model Dataset
// struct LMDataset {
//     std::vector<std::vector<int>> sentences;
//     Vocabulary vocab;
//     const int max_len = 70;
//
//     void load_data(const std::string& filename) {
//         std::ifstream file(filename);
//         std::string line;
//         std::vector<std::string> raw_sentences;
//         while (std::getline(file, line)) {
//             if (!line.empty()) {
//                 raw_sentences.push_back(line);
//             }
//         }
//         vocab.build(raw_sentences);
//         for (const auto& sentence : raw_sentences) {
//             auto tokens = vocab.tokenize(sentence);
//             if (tokens.size() > max_len) {
//                 tokens.resize(max_len);
//             }
//             sentences.push_back(tokens);
//         }
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//         std::vector<torch::Tensor> input_batch, target_batch, lengths_batch;
//         size_t max_seq_len = 0;
//
//         for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
//             max_seq_len = std::max(max_seq_len, sentences[i].size());
//         }
//
//         for (size_t i = idx; i < std::min(idx + batch_size, sentences.size()); ++i) {
//             auto& tokens = sentences[i];
//             std::vector<int> input_tokens = tokens;
//             std::vector<int> target_tokens(tokens.begin() + 1, tokens.end());
//             target_tokens.push_back(vocab.pad_id); // Shift for LM
//             while (input_tokens.size() < max_seq_len) {
//                 input_tokens.push_back(vocab.pad_id);
//                 target_tokens.push_back(vocab.pad_id);
//             }
//             input_batch.push_back(torch::tensor(input_tokens, torch::kInt64));
//             target_batch.push_back(torch::tensor(target_tokens, torch::kInt64));
//             lengths_batch.push_back(torch::tensor(static_cast<int64_t>(tokens.size()), torch::kInt64));
//         }
//
//         return {
//             torch::stack(input_batch),
//             torch::stack(target_batch),
//             torch::stack(lengths_batch)
//         };
//     }
// };
//
// // Classification Dataset
// struct ClassificationDataset {
//     std::vector<std::pair<std::string, int>> labeled_sentences;
//     Vocabulary vocab;
//     const int max_len = 70;
//
//     void load_data(const std::string& filename) {
//         std::ifstream file(filename);
//         std::string line;
//         std::vector<std::string> sentences;
//         while (std::getline(file, line)) {
//             std::istringstream iss(line);
//             int label;
//             iss >> label;
//             std::string sentence;
//             std::getline(iss, sentence);
//             sentence.erase(0, sentence.find_first_not_of(" \t"));
//             if (!sentence.empty()) {
//                 labeled_sentences.push_back({sentence, label});
//                 sentences.push_back(sentence);
//             }
//         }
//         vocab.build(sentences);
//     }
//
//     std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> get_batch(size_t idx, size_t batch_size) {
//         std::vector<torch::Tensor> input_batch, label_batch, lengths_batch;
//         size_t max_seq_len = 0;
//
//         for (size_t i = idx; i < std::min(idx + batch_size, labeled_sentences.size()); ++i) {
//             auto tokens = vocab.tokenize(labeled_sentences[i].first);
//             max_seq_len = std::max(max_seq_len, std::min(tokens.size(), static_cast<size_t>(max_len)));
//         }
//
//         for (size_t i = idx; i < std::min(idx + batch_size, labeled_sentences.size()); ++i) {
//             auto& [sentence, label] = labeled_sentences[i];
//             auto tokens = vocab.tokenize(sentence);
//             if (tokens.size() > max_len) {
//                 tokens.resize(max_len);
//             }
//             while (tokens.size() < max_seq_len) {
//                 tokens.push_back(vocab.pad_id);
//             }
//             input_batch.push_back(torch::tensor(tokens, torch::kInt64));
//             label_batch.push_back(torch::tensor(label, torch::kInt64));
//             lengths_batch.push_back(torch::tensor(static_cast<int64_t>(std::min(tokens.size(), static_cast<size_t>(max_len))), torch::kInt64));
//         }
//
//         return {
//             torch::stack(input_batch),
//             torch::stack(label_batch),
//             torch::stack(lengths_batch)
//         };
//     }
// };
//
// // Slanted triangular learning rate scheduler
// class SlantedTriangularLR {
// public:
//     SlantedTriangularLR(torch::optim::Optimizer& optimizer, int max_steps, float max_lr, float cut_frac = 0.1)
//         : optimizer_(optimizer), max_steps_(max_steps), max_lr_(max_lr), cut_frac_(cut_frac) {}
//
//     void step(int step) {
//         float frac = static_cast<float>(step) / max_steps_;
//         float cut = cut_frac_;
//         float lr;
//         if (frac < cut) {
//             lr = max_lr_ * (frac / cut);
//         } else {
//             lr = max_lr_ * (1 - (frac - cut) / (1 - cut));
//         }
//         for (auto& group : optimizer_.param_groups()) {
//             group.options().set_lr(lr);
//         }
//     }
//
// private:
//     torch::optim::Optimizer& optimizer_;
//     int max_steps_;
//     float max_lr_;
//     float cut_frac_;
// };
//
// int main() {
//     torch::manual_seed(0);
//     srand(static_cast<unsigned>(time(0)));
//
//     // Load datasets
//     LMDataset lm_dataset;
//     lm_dataset.load_data("lm_corpus.txt"); // General corpus for LM fine-tuning
//     ClassificationDataset cls_dataset;
//     cls_dataset.load_data("cls_data.txt"); // Labeled data for classification
//
//     // Initialize model
//     ULMFiT model(lm_dataset.vocab.vocab_size, 200, 400, 2, 2); // 2 classes: positive/negative
//     torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
//     model.to(device);
//
//     // Optimizer
//     torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));
//
//     // Language Model Fine-tuning
//     model.set_classifier_mode(false);
//     const int lm_epochs = 5;
//     const size_t lm_batch_size = 64;
//     auto lm_criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().ignore_index(lm_dataset.vocab.pad_id));
//     SlantedTriangularLR lm_scheduler(optimizer, lm_epochs * (lm_dataset.sentences.size() / lm_batch_size), 0.01);
//
//     for (int epoch = 0; epoch < lm_epochs; ++epoch) {
//         float total_loss = 0.0;
//         size_t num_batches = 0;
//         model.train();
//
//         for (size_t i = 0; i < lm_dataset.sentences.size(); i += lm_batch_size) {
//             auto [input, target, lengths] = lm_dataset.get_batch(i, lm_batch_size);
//             input = input.to(device);
//             target = target.to(device);
//             lengths = lengths.to(device);
//
//             optimizer.zero_grad();
//             auto output = model.forward(input, lengths); // [batch, seq_len, vocab_size]
//             auto loss = lm_criterion->forward(output.view({-1, lm_dataset.vocab.vocab_size}), target.view(-1));
//             loss.backward();
//             torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
//             optimizer.step();
//             lm_scheduler.step(epoch * (lm_dataset.sentences.size() / lm_batch_size) + (i / lm_batch_size));
//
//             total_loss += loss.item<float>();
//             num_batches++;
//         }
//
//         std::cout << "LM Fine-tuning Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
//     }
//
//     // Classifier Training
//     model.set_classifier_mode(true);
//     const int cls_epochs = 10;
//     const size_t cls_batch_size = 32;
//     auto cls_criterion = torch::nn::CrossEntropyLoss();
//     SlantedTriangularLR cls_scheduler(optimizer, cls_epochs * (cls_dataset.labeled_sentences.size() / cls_batch_size), 0.002);
//
//     // Gradual unfreezing: freeze all but classifier initially
//     for (auto& param : model.parameters()) {
//         param.set_requires_grad(false);
//     }
//     for (auto& param : model.classifier->parameters()) {
//         param.set_requires_grad(true);
//     }
//
//     for (int epoch = 0; epoch < cls_epochs; ++epoch) {
//         float total_loss = 0.0;
//         size_t num_batches = 0;
//         model.train();
//
//         // Unfreeze more layers after a few epochs
//         if (epoch == 2) {
//             for (auto& param : model.encoder->parameters()) {
//                 param.set_requires_grad(true);
//             }
//         }
//         if (epoch == 4) {
//             for (auto& param : model.embedding->parameters()) {
//                 param.set_requires_grad(true);
//             }
//         }
//
//         for (size_t i = 0; i < cls_dataset.labeled_sentences.size(); i += cls_batch_size) {
//             auto [input, labels, lengths] = cls_dataset.get_batch(i, cls_batch_size);
//             input = input.to(device);
//             labels = labels.to(device);
//             lengths = lengths.to(device);
//
//             optimizer.zero_grad();
//             auto output = model.forward(input, lengths); // [batch, num_classes]
//             auto loss = cls_criterion->forward(output, labels);
//             loss.backward();
//             torch::nn::utils::clip_grad_norm_(model.parameters(), 1.0);
//             optimizer.step();
//             cls_scheduler.step(epoch * (cls_dataset.labeled_sentences.size() / cls_batch_size) + (i / cls_batch_size));
//
//             total_loss += loss.item<float>();
//             num_batches++;
//         }
//
//         std::cout << "Classifier Epoch: " << epoch + 1 << ", Loss: " << total_loss / num_batches << std::endl;
//     }
//
//     // Save model
//     torch::save(model, "ulmfit_model.pt");
//
//     return 0;
// }


namespace xt::models
{
    ULMFiT::ULMFiT(int num_classes, int in_channels)
    {
    }

    ULMFiT::ULMFiT(int num_classes, int in_channels, std::vector<int64_t> input_shape)
    {
    }

    void ULMFiT::reset()
    {
    }

    auto ULMFiT::forward(std::initializer_list<std::any> tensors) -> std::any
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
