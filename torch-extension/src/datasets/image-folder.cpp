//
// #include "../../include/datasets/base.h"
//
//
// // Partially based on pytorch ImageFolder dataset
// // See Pytorch Python implementation: https://pytorch.org/vision/stable/_modules/torchvision/datasets/folder.html#ImageFolder
// #include <torch/torch.h>
// #include <vector>
// #include <algorithm>
// //#include "image_io.h"
// #include <filesystem>
// #include <unordered_map>
//
// namespace fs = std::filesystem;
//
// //using image_io::load_image;
//
// namespace dataset
// {
//     namespace
//     {
//         std::vector<std::string> parse_classes(const std::string &directory)
//         {
//             std::vector<std::string> classes;
//
//             for (auto &p : fs::directory_iterator(directory))
//             {
//
//                 if (p.is_directory())
//                 {
//                     classes.push_back(p.path().filename().string());
//                 }
//             }
//
//             std::sort(classes.begin(), classes.end());
//
//             return classes;
//         }
//
//         std::unordered_map<std::string, int> create_class_to_index_map(const std::vector<std::string> &classes)
//         {
//             std::unordered_map<std::string, int> class_to_index;
//
//             int index = 0;
//
//             for (const auto &class_name : classes)
//             {
//                 class_to_index[class_name] = index++;
//             }
//
//             return class_to_index;
//         }
//
//         std::vector<std::pair<std::string, int>> create_samples(const std::string &directory,
//                                                                 const std::unordered_map<std::string, int> &class_to_index)
//         {
//             std::vector<std::pair<std::string, int>> samples;
//
//             for (const auto &[class_name, class_index] : class_to_index)
//             {
//                 for (const auto &p : fs::directory_iterator(directory + "/" + class_name))
//                 {
//                     if (p.is_regular_file())
//                     {
//                         samples.emplace_back(p.path().string(), class_index);
//                     }
//                 }
//             }
//
//             return samples;
//         }
//     } // namespace
//
//     ImageFolder::ImageFolder(const std::string &root, Mode mode, torch::IntArrayRef image_load_size)
//             : mode_(mode),
//               image_load_size_(image_load_size.begin(), image_load_size.end()),
//               mode_dir_(root + "/" + (mode == Mode::TRAIN ? "train" : "val")),
//               classes_(parse_classes(mode_dir_)),
//               class_to_index_(create_class_to_index_map(classes_)),
//               samples_(create_samples(mode_dir_, class_to_index_))
//     {
//     }
//
//     torch::optional<size_t> ImageFolder::size() const
//     {
//         return samples_.size();
//     }
//
//     torch::data::Example<> ImageFolder::get(size_t index)
//     {
//         const auto &[image_path, class_index] = samples_[index];
//
// //        return {load_image(image_path, image_load_size_), torch::tensor(class_index)};
//         return {};
//     }
// } // namespace dataset
//
//
// //#include <torch/torch.h>
// //#include <iostream>
// //#include "imagefolder_dataset.h"
// //
// //using dataset::ImageFolder;
// //
// //namespace fs = std::filesystem;
// //
// //int main(int argc, char **argv)
// //{
// //    // Assumes structure: <root_directory>/(train|val)/<class>/<img_file>
// //    auto root_directory = "/abs/path/to/dataset/root";
// //
// //    ImageFolderDataset imagenet_dataset(root_directory, ImageFolderDataset::Mode::TRAIN,
// //                                        {160, 160});
// //
// //    auto example = imagenet_dataset.get(0);
// //
// //    std::cout << example.data.sizes() << ", " << example.target << std::endl;
// //
// //    // ...
// //}
