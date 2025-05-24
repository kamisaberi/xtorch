#include "include/datasets/computer_vision/image_classification/food.h"

namespace xt::data::datasets
{
    Food101::Food101(const std::string& root): Food101(root, xt::datasets::DataMode::TRAIN, false, nullptr, nullptr)
    {
    }

    Food101::Food101(const std::string& root, xt::datasets::DataMode mode) : Food101(
        root, mode, false, nullptr, nullptr)
    {
    }

    Food101::Food101(const std::string& root, xt::datasets::DataMode mode, bool download) : Food101(
        root, mode, download, nullptr, nullptr)
    {
    }

    Food101::Food101(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer): Food101(root, mode, download, std::move(transformer),
                                                                       nullptr)
    {
    }

    Food101::Food101(const std::string& root, xt::datasets::DataMode mode, bool download,
                     std::unique_ptr<xt::Module> transformer,
                     std::unique_ptr<xt::Module> target_transformer) : xt::datasets::Dataset(
        mode, std::move(transformer), std::move(target_transformer))
    {
        check_resources(); // Verify dataset files exist
        load_classes(); // Load class names and mapping
        load_data(); // Load image data and labels
    }

    void Food101::check_resources()
    {
        fs::path archive_file_abs_path = this->root / this->dataset_file_name;
        this->dataset_path = this->root / this->dataset_folder_name;
        fs::path images_path = this->dataset_path / fs::path("images");

        // Check if extracted dataset exists
        if (fs::exists(images_path))
        {
            size_t cnt = xt::utils::fs::countFiles(images_path, true);
            // Verify complete dataset
            if (cnt != this->images_number)
            {
                if (fs::exists(archive_file_abs_path))
                {
                    xt::utils::extract(archive_file_abs_path, this->root.string());
                }
                else
                {
                    xt::utils::download(this->url, archive_file_abs_path.string());
                    xt::utils::extract(archive_file_abs_path, this->root.string());
                }
            }
        }
        else
        {
            // Handle case where dataset not extracted
            if (fs::exists(archive_file_abs_path))
            {
                xt::utils::extract(archive_file_abs_path, this->root.string());
            }
            else
            {
                // Full download and extract
                xt::utils::download(this->url, archive_file_abs_path.string());
                xt::utils::extract(archive_file_abs_path, this->root.string());
            }
        }
    }

    void Food101::load_classes()
    {
        cout << "Loading classes..." << endl;
        fs::path meta_path = this->dataset_path / fs::path("meta");
        fs::path classes_file_path = meta_path / fs::path("classes.txt");

        // Read class names line by line
        ifstream ifs(classes_file_path.string());
        while (!ifs.eof())
        {
            std::string line;
            std::getline(ifs, line);
            line = xt::utils::string::trim(line);
            if (line.empty())
                continue;

            // Add to name vector and create mapping
            classes_name.push_back(line);
            classes_map.insert({line, classes_name.size() - 1});
        }
    }

    void Food101::load_data()
    {
        fs::path images_path = this->dataset_path / fs::path("images");
        fs::path meta_path = this->dataset_path / fs::path("meta");

        // Handle training set
        if (this->mode == xt::datasets::DataMode::TRAIN)
        {
            fs::path train_file_path = meta_path / fs::path("train.txt");
            ifstream ifs(train_file_path.string());

            while (!ifs.eof())
            {
                string line;
                getline(ifs, line);
                line = xt::utils::string::trim(line);
                if (line.empty())
                    continue;

                // Parse line format: "class_name/image_name"
                vector<string> tokens = xt::utils::string::split(line, "/");
                string label = tokens[0];

                // Load and process image
                fs::path img_path = images_path / fs::path(line + ".jpg");
                torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img_path);

                // Apply transforms if specified
                if (transformer != nullptr)
                {
                    tensor = (*transformer)(tensor);
                }

                // Store results
                this->data.push_back(tensor);
                this->targets.push_back(classes_map[label]);
            }
        }
        // Handle test set
        else
        {
            fs::path train_file_path = meta_path / fs::path("test.txt");
            ifstream ifs(train_file_path.string());

            while (!ifs.eof())
            {
                string line;
                getline(ifs, line);
                line = xt::utils::string::trim(line);
                if (line.empty())
                    continue;

                // Parse line format: "class_name/image_name"
                vector<string> tokens = xt::utils::string::split(line, "/");
                string label = tokens[0];

                // Load and process image
                fs::path img_path = images_path / fs::path(line + ".jpg");
                torch::Tensor tensor = torch::ext::media::opencv::convertImageToTensor(img_path);

                // Apply transforms if specified
                if (transformer != nullptr)
                {
                    tensor = (*transformer)(tensor);
                }

                // Store results
                this->data.push_back(tensor);
                this->targets.push_back(classes_map[label]);
            }
        }
    }
}
