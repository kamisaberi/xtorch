# Image Generation: DCGAN on CelebA

This tutorial dives into the exciting field of generative modeling by demonstrating how to train a **Deep Convolutional Generative Adversarial Network (DCGAN)**. Our goal is to train a model that can generate novel, realistic images of celebrity faces.

We will be using the **CelebA dataset** and the pre-built `xt::models::DCGAN` architecture provided by xTorch.

This example is more advanced than a standard classification task because it involves:
1.  Managing two separate models: a **Generator** and a **Discriminator**.
2.  Implementing a custom training loop where these two models compete against each other.
3.  Using a specific loss function (`BCELoss`) and optimizer configuration (`Adam` with `beta1=0.5`) that are known to work well for GANs.

---

## The GAN Training Process

A Generative Adversarial Network is trained as a zero-sum game between two competing neural networks:

-   **The Generator (`G`)**: Its job is to create realistic-looking images from random noise. It starts by producing garbage but gets better over time.
-   **The Discriminator (`D`)**: Its job is to act as a detective, trying to distinguish between "real" images (from the CelebA dataset) and "fake" images created by the Generator.

The training loop alternates between these two players.

#### Step 1: Train the Discriminator

The Discriminator is a standard binary classifier. Its training is done in two parts:
1.  **Real Batch**: We show it a batch of real images from the dataset and teach it to classify them as "real" (label = 1).
2.  **Fake Batch**: The Generator creates a batch of fake images. We show these to the Discriminator and teach it to classify them as "fake" (label = 0).

The gradients from both parts are combined, and the Discriminator's weights are updated.

#### Step 2: Train the Generator

The Generator's goal is to fool the Discriminator.
1.  The Generator creates a batch of fake images.
2.  These fake images are passed to the Discriminator.
3.  We calculate the Generator's loss based on how well it tricked the Discriminator. Specifically, the Generator wants the Discriminator to classify its fake images as "real" (label = 1).
4.  We then compute gradients and update **only the Generator's weights**.

This adversarial process forces the Generator to produce increasingly realistic images to keep up with the improving Discriminator.

---

## Full C++ Code

Below is the complete source code for training the DCGAN. The original file can be found at `computer_vision/image_generation/generating_images_with_dcgan.cpp`.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>
#include <chrono>

int main() {
    try {
        // --- 1. Hyperparameters ---
        const int latent_vector_size = 100; // Size of input noise vector
        const int generator_feature_maps = 64;
        const int discriminator_feature_maps = 64;
        const int num_channels = 3;
        const int num_epochs = 5;
        const int batch_size = 128;
        const double lr = 0.0002;
        const double beta1 = 0.5;
        const std::vector<int64_t> image_size = {64, 64};

        // --- 2. Setup Device, Models, and Optimizers ---
        torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
        std::cout << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << std::endl;

        // Initialize Generator and Discriminator from the xTorch model zoo
        xt::models::DCGAN::Generator netG(latent_vector_size, generator_feature_maps, num_channels);
        xt::models::DCGAN::Discriminator netD(num_channels, discriminator_feature_maps);
        netG.to(device);
        netD.to(device);

        // Setup Adam optimizers for both models
        torch::optim::Adam optimG(netG.parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));
        torch::optim::Adam optimD(netD.parameters(), torch::optim::AdamOptions(lr).betas({beta1, 0.999}));

        // Loss function
        torch::nn::BCELoss criterion;

        // --- 3. Data Pipeline ---
        auto transforms = std::make_unique<xt::transforms::Compose>(
            std::make_shared<xt::transforms::image::Resize>(image_size),
            std::make_shared<xt::transforms::image::CenterCrop>(image_size),
            // Normalize images to the [-1, 1] range, typical for GANs
            std::make_shared<xt::transforms::general::Normalize>({0.5, 0.5, 0.5}, {0.5, 0.5, 0.5})
        );

        auto dataset = xt::datasets::CelebA(
            "/path/to/your/datasets/", // IMPORTANT: Change this path
            xt::datasets::DataMode::TRAIN,
            /*download=*/true,
            std::move(transforms)
        );
        xt::dataloaders::ExtendedDataLoader data_loader(dataset, batch_size, true, 4, 2);

        // --- 4. The GAN Training Loop ---
        std::cout << "\nStarting GAN training loop..." << std::endl;
        auto start_time = std::chrono::steady_clock::now();

        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            int batch_idx = 0;
            for (auto& batch : data_loader) {
                // ------------ Part 1: Train the Discriminator ------------
                netD.zero_grad();
                // 1a. Train with a real batch
                auto real_data = batch.first.to(device);
                auto current_batch_size = real_data.size(0);
                auto real_labels = torch::full({current_batch_size}, 1.0, torch::kFloat).to(device);

                auto output = torch::sigmoid(netD.forward(real_data)).view(-1);
                auto errD_real = criterion(output, real_labels);
                errD_real.backward();

                // 1b. Train with a fake batch
                auto noise = torch::randn({current_batch_size, latent_vector_size, 1, 1}).to(device);
                auto fake_data = netG.forward(noise);
                auto fake_labels = torch::full({current_batch_size}, 0.0, torch::kFloat).to(device);

                output = torch::sigmoid(netD.forward(fake_data.detach())).view(-1);
                auto errD_fake = criterion(output, fake_labels);
                errD_fake.backward();

                // Update the discriminator
                auto errD = errD_real + errD_fake;
                optimD.step();

                // ------------ Part 2: Train the Generator ------------
                netG.zero_grad();
                // The generator's goal is to make the discriminator think its fakes are real (label=1)
                output = torch::sigmoid(netD.forward(fake_data)).view(-1);
                auto errG = criterion(output, real_labels);
                errG.backward();

                // Update the generator
                optimG.step();

                if (++batch_idx % 50 == 0) {
                    std::cout << "Epoch [" << epoch + 1 << "/" << num_epochs << "] Batch [" << batch_idx << "/"
                              << *dataset.size() / batch_size << "] D_Loss: " << errD.item<float>()
                              << " G_Loss: " << errG.item<float>() << std::endl;
                }
            }
        }
        auto end_time = std::chrono::steady_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        std::cout << "\nTotal training duration: " << duration_ms.count() << " milliseconds." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
```

## How to Compile and Run

This example can be found in the `xtorch-examples` repository.
1.  Navigate to the `computer_vision/image_generation/` directory.
2.  Build using CMake:
    ```bash
    mkdir build
    cd build
    cmake ..
    make
    ```
3.  Run the executable:
    ```bash
    ./generate_images_dcgan
    ```

## Expected Output

You will see the progress of the training loop, with the Discriminator Loss (`D_Loss`) and Generator Loss (`G_Loss`) printed to the console. Ideally, these two losses should remain in a rough equilibrium; if one drops to zero, the other network stops learning.

```
Using device: CUDA
Starting GAN training loop...
Epoch [1/5] Batch [50/1583] D_Loss: 0.5432109833 G_Loss: 3.1234567890
Epoch [1/5] Batch [100/1583] D_Loss: 0.4321098765 G_Loss: 3.5678901234
...
Total training duration: 219000 milliseconds.
```

At the end of training, the generator (`netG`) will be capable of producing novel images of faces from random noise vectors. You could modify the code to save some generated image samples to disk to visualize the results.