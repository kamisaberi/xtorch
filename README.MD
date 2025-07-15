![Logo](https://github.com/user-attachments/assets/70527c02-c73e-429b-9d86-0b43172dccb2)

# 🔴 _LIBRARY UNDER DEVELOPMENT SITUATION PLEASE ONLY USE RELEASE VERSION_

[//]: # (# xTorch: Bridging the Usability Gap in PyTorch’s C++ API)

## Motivation

PyTorch’s C++ library (LibTorch) emerged as a powerful way to use PyTorch outside Python, but after 2019 it became
challenging for developers to use it for end-to-end model development. Early on, LibTorch aimed to mirror the high-level
Python API, yet many convenient abstractions and examples never fully materialized or were later removed.

As of 2020, the C++ API had achieved near feature-parity with Python’s core operations, but it lagged in usability and
community support. Fewer contributors focused on C++ meant that only low-level building blocks were provided, with
high-level components (e.g. ready-made network architectures, datasets) largely absent. This left C++ practitioners to
rewrite common tools from scratch – implementing standard models or data loaders manually – which is time-consuming and
error-prone.

Another factor was PyTorch’s emphasis on the Python-to-C++ workflow. The official recommended path for production was to
prototype in Python, then convert models to TorchScript for C++ deployment. This approach deprioritized making the pure
C++ experience as friendly as Python’s.

As a result, developers who preferred or needed to work in C++ (for integration with existing systems, performance, or
deployment constraints) found LibTorch cumbersome. Simple tasks like data augmentation (e.g. random crops or flips) had
no built-in support in LibTorch C++. Defining neural network modules in C++ involved boilerplate macros and manual
registration, an awkward process compared to Python’s concise syntax. Crucial functionality for model serialization was
limited – for instance, LibTorch could load Python-exported models but not easily export its own models to a portable
format.

xTorch was created to address this gap. It is a C++ library that extends LibTorch with the high-level abstractions and
utilities that were missing or removed after 2019. By building on LibTorch’s robust computational core, xTorch restores
ease-of-use without sacrificing performance. The motivation is to empower C++ developers with a productive experience
similar to PyTorch in Python – enabling them to build, train, and deploy models with minimal fuss. In essence, xTorch
revives and modernizes the “batteries-included” ethos for C++ deep learning, providing an all-in-one toolkit where the
base library left off.

## Design and Architecture

xTorch is architected as a thin layer on top of LibTorch’s C++ API, carefully integrating with it rather than
reinventing it. The design follows a modular approach, adding a higher-level API that wraps around LibTorch’s
lower-level classes. At its core, xTorch relies on LibTorch for tensor operations, autograd, and neural network
primitives – effectively using LibTorch as the computational engine. The extended library then introduces its own set of
C++ classes that encapsulate common patterns (model definitions, training loops, data handling, etc.), providing a
cleaner interface to the developer.

### Architecture Layers

- **LibTorch Core (Bottom Layer):** Provides `torch::Tensor`, `torch::autograd`, `torch::nn`, optimizers, etc.
- **Extended Abstraction Layer (Middle):** Simplified classes inheriting from LibTorch core (e.g., `ExtendedModel`,
  `Trainer`).
- **User Interface (Top Layer):** Intuitive APIs and boilerplate-free interaction.

### Modules

- **Model Module:** High-level model class extensions.
- **Data Module:** Enhanced datasets and DataLoader.
- **Training Module:** Training logic, checkpointing, metrics.
- **Utilities Module:** Logging, device helpers, summaries.

## Features and Enhancements

- **High-Level Model Classes:** `XTModule`, prebuilt models like `ResNetExtended`, `XTCNN`.
- **Simplified Training Loop (Trainer):** Full training abstraction with callbacks and metrics.
- **Enhanced Data Handling:** `ImageFolderDataset`, `CSVDataset`, OpenCV-backed support.
- **Utility Functions:** Logging, metrics, summary, device utils.
- **Extended Optimizers:** AdamW, RAdam, schedulers, learning rate strategies.
- **Model Serialization & Deployment:** `save_model()`, `export_to_jit()`, inference helpers.

## Use Cases and Examples

### Example: CNN Training Pipeline (Simplified)

```cpp
#include <xtorch/xtorch.h>
#include <iostream>


int main() {
    std::cout.precision(10);
    std::vector<std::shared_ptr<xt::Module>> transform_list;
    transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(std::vector<int64_t>{32, 32}));
    transform_list.push_back(
        std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5}, std::vector<float>{0.5}));
    auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
    auto dataset = xt::datasets::MNIST("/home/kami/Documents/datasets/", xt::datasets::DataMode::TRAIN, false,
                                       std::move(compose));
    xt::dataloaders::ExtendedDataLoader data_loader(dataset, 64, true, 2, /*prefetch_factor=*/2);
    xt::models::LeNet5 model(10);
    model.to(torch::Device(torch::kCPU));
    model.train();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(1e-3));
    auto logger = std::make_shared<xt::LoggingCallback>("[MyTrain]", /*log_every_N_batches=*/20, /*log_time=*/true);
    xt::Trainer trainer;
    trainer.set_max_epochs(10).set_optimizer(optimizer)
           .set_loss_fn([](const auto& output, const auto& target)
           {
               return torch::nll_loss(output, target);
           })
           .add_callback(logger);
    trainer.fit(model, data_loader, &data_loader, torch::Device(torch::kCPU));
    return 0;
}
```

### Example: DCGAN Generative AI

```cpp
#include <xtorch/xtorch.h>
#include <iostream>


int main() {
    try
    {
        // Hyperparameters
        const int nz = 100; // Size of latent vector
        const int ngf = 64; // Size of feature maps in generator
        const int ndf = 64; // Size of feature maps in discriminator
        const int nc = 3; // Number of channels (1 for MNIST)
        const int num_epochs = 5;
        const int batch_size = 128;
        const double lr = 0.0002;
        const double beta1 = 0.5;
        const vector<int64_t> image_size = {64, 64};
        const std::string dataroot = "./data";

        // Device
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // device = torch::Device(torch::kCPU);
        // Initialize models
        xt::models::DCGAN::Generator netG(nz, ngf, nc);
        xt::models::DCGAN::Discriminator netD(nc, ndf);
        netG.to(device);
        netD.to(device);

        // Optimizers (fixed: optimD uses netD.parameters())
        torch::optim::Adam optimG(netG.parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));
        torch::optim::Adam optimD(netD.parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));

        // Loss function
        torch::nn::BCELoss criterion;

        // Transforms (adjusted normalization for MNIST to [-1, 1])
        std::vector<std::shared_ptr<xt::Module>> transform_list;
        transform_list.push_back(std::make_shared<xt::transforms::image::Resize>(image_size));
        transform_list.push_back(std::make_shared<xt::transforms::image::CenterCrop>(image_size));
        transform_list.push_back(
            std::make_shared<xt::transforms::general::Normalize>(std::vector<float>{0.5, 0.5, 0.5,},
                                                                 std::vector<float>{0.5, 0.5, 0.5}));


        auto compose = std::make_unique<xt::transforms::Compose>(transform_list);
        auto dataset = xt::datasets::CelebA("/home/kami/Documents/datasets/",
                                            xt::datasets::DataMode::TRAIN, false,
                                            std::move(compose));
        cout << dataset.size().value() << endl;
        cout << dataset.get(0).data.sizes() << endl;
        cout << dataset.get(1).data.sizes() << endl;
        cout << dataset.get(2).data.sizes() << endl;
        xt::dataloaders::ExtendedDataLoader data_loader(dataset, batch_size, true, 2, /*prefetch_factor=*/2);


        auto start_time = std::chrono::steady_clock::now();

        // return 0;
        for (int epoch = 0; epoch < num_epochs; ++epoch)
        {
            int i = 1;
            for (auto& batch : data_loader)
            {
                netD.zero_grad();
                auto real_data = batch.first.to(device);

                auto batch_size = real_data.size(0);

                auto real_label = torch::full({batch_size}, 1.0,
                                              torch::TensorOptions().device(device).dtype(torch::kFloat));
                auto output = torch::sigmoid(netD.forward(real_data)).view(-1); // Added sigmoid for BCELoss
                auto errD_real = criterion(output, real_label);
                errD_real.backward();

                auto noise = torch::randn({batch_size, nz, 1, 1}, torch::TensorOptions().device(device));
                auto fake_data = netG.forward(noise);
                auto fake_label = torch::full({batch_size}, 0.0,
                                              torch::TensorOptions().device(device).dtype(torch::kFloat));
                output = torch::sigmoid(netD.forward(fake_data.detach())).view(-1); // Added sigmoid
                auto errD_fake = criterion(output, fake_label);
                errD_fake.backward();
                auto errD = errD_real + errD_fake;
                optimD.step();

                netG.zero_grad();
                output = torch::sigmoid(netD.forward(fake_data)).view(-1); // Added sigmoid
                auto errG = criterion(output, real_label); // Generator wants discriminator to think fakes are real
                errG.backward();
                optimG.step();

                // Print progress
                if (i % 50 == 0)
                    std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] D_Loss: " << errD.item<float>() <<
                        " G_Loss: " << errG.item<float>() << " -- " << i << " of " << dataset.size().value() /
                        batch_size
                        << std::endl;
                i++;
            }

            if (epoch % 2 == 0)
            {
                auto noise = torch::randn({16, nz, 1, 1}, torch::TensorOptions().device(device));
                auto fake_images = netG.forward(noise);
            }
        }
        auto end_time = std::chrono::steady_clock::now();
        auto duration = end_time - start_time;
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
        std::cout << "Total loop duration: " << duration_ms.count() << " milliseconds." << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### Example: C++ Inference Pipeline

```cpp
auto model = xt::load_model("resnet18_script.pt");
auto tensor = xt::utils::imageToTensor("input.jpg");
auto outputs = xt::utils::predict(model, tensor);
int predictedClass = xt::utils::argmax(outputs);
std::cout << "Predicted class = " << predictedClass << std::endl;
```

## Performance

XTorch was born from a simple observation: Python's inherent performance bottlenecks are a major liability in
data-intensive AI workloads. By re-architecting the deep learning stack in native C++, XTorch eliminates these
limitations. In a direct benchmark training a DCGAN on CelebA, XTorch on a **single NVIDIA RTX 3090 GPU** completed a
5-epoch run in **219 seconds**. This represents a **37.4% speedup** over a standard PyTorch implementation which
required **350 seconds** using **two RTX 3090 GPUs** with `DataParallel`. This result demonstrates that a
performance-first, C++ native approach can not only match but significantly outperform common multi-GPU Python setups,
enabling faster iteration and reducing hardware costs.

## Impact and Potential Applications

- **C++ Developers:** Enables use of PyTorch-like training without Python.
- **Research in Embedded / HPC:** Pure C++ training and deployment possible.
- **Industrial Use:** On-device training, edge deployment workflows.
- **Education:** Useful for teaching performance-aware ML in C++.
- **Ecosystem Growth:** Boosts community contributions, reuse, and experimentation.

## Comparison with Related Tools

| Feature                    | LibTorch | xTorch | PyTorch Lightning (Python) |
|----------------------------|----------|--------|----------------------------|
| Training Loop Abstraction  | ❌        | ✅      | ✅                          |
| Data Augmentation Built-in | ❌        | ✅      | ✅                          |
| Built-in Model Zoo         | Limited  | ✅      | ✅                          |
| Target Language            | C++      | C++    | Python                     |
| TorchScript Export         | Limited  | ✅      | ✅                          |

xTorch complements PyTorch’s C++ API like PyTorch Lightning does in Python, enabling expressive ML development in C++
with clean, modular code structures.

