# Generative Models

Generative models are a fascinating class of neural networks that learn to create new data samples that resemble the training data. They can be used for a wide range of creative and practical applications, including image synthesis, data augmentation, and unsupervised feature learning.

xTorch provides implementations of several major families of generative models, allowing you to easily experiment with this cutting-edge field.

All generative models are located under the `xt::models` namespace and their headers can be found in the `<xtorch/models/generative_models/>` directory.

## General Usage

The usage of generative models often involves more complex training loops than standard supervised learning. For example, training a Generative Adversarial Network (GAN) requires managing two separate models (a Generator and a Discriminator) and their respective optimizers.

The example below shows how to instantiate the Generator and Discriminator for a DCGAN.

!!! note "Training Generative Models"
    Due to their often complex training dynamics (e.g., alternating between training a generator and a discriminator), the standard `xt::Trainer` might not be suitable for all generative models out-of-the-box. Many of the examples for these models will feature a custom training loop.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

    // --- Hyperparameters for a DCGAN ---
    const int latent_vector_size = 100; // Size of the input noise vector (nz)
    const int generator_feature_maps = 64; // Size of feature maps in generator (ngf)
    const int discriminator_feature_maps = 64; // Size of feature maps in discriminator (ndf)
    const int num_channels = 3; // Number of image channels (nc)

    // --- Instantiate the Generator and Discriminator ---
    xt::models::DCGAN::Generator generator(
        latent_vector_size,
        generator_feature_maps,
        num_channels
    );
    generator.to(device);

    xt::models::DCGAN::Discriminator discriminator(
        num_channels,
        discriminator_feature_maps
    );
    discriminator.to(device);

    std::cout << "DCGAN Generator and Discriminator instantiated." << std::endl;

    // --- Perform a dummy forward pass ---
    // Create a random noise tensor to feed the generator
    auto noise = torch::randn({16, latent_vector_size, 1, 1}).to(device);
    // Generate a batch of fake images
    auto fake_images = generator.forward(noise);

    // Pass the fake images to the discriminator
    auto discriminator_output = discriminator.forward(fake_images);

    std::cout << "Generated image batch shape: " << fake_images.sizes() << std::endl;
    std::cout << "Discriminator output shape: " << discriminator_output.sizes() << std::endl;

    // --- Setup Optimizers (for a real training loop) ---
    torch::optim::Adam generator_optimizer(generator.parameters(), torch::optim::AdamOptions(0.0002).betas({0.5, 0.999}));
    torch::optim::Adam discriminator_optimizer(discriminator.parameters(), torch::optim::AdamOptions(0.0002).betas({0.5, 0.999}));
}
```

---

## Available Models by Family

### Generative Adversarial Networks (GANs)

GANs consist of a generator that creates data and a discriminator that tries to distinguish between real and generated data.

| Model Family | Description | Header File |
|---|---|---|
| `GAN` | A basic, foundational GAN implementation. | `gans/gan.h` |
| `DCGAN` | Deep Convolutional GAN, a stable and effective architecture for image generation. | `gans/dcgan.h` |
| `WGAN` | Wasserstein GAN, which uses the Wasserstein distance to improve training stability. | `gans/wgan.h` |
| `WGAN-GP`| WGAN with a Gradient Penalty, further improving stability over the original WGAN. | `gans/wgan_gp.h` |
| `CycleGAN` | A model for unpaired image-to-image translation. | `gans/cycle_gan.h` |
| `Pix2Pix` | A model for paired image-to-image translation. | `gans/pix2pix.h` |
| `ProGAN` | Progressively Growing GANs, for generating high-resolution images. | `gans/pro_gan.h` |
| `StyleGAN` | A powerful GAN architecture that allows for style-based control over the generated images. | `gans/style_gan.h` |
| `BigGAN` | A large-scale GAN known for generating high-fidelity and diverse images. | `gans/big_gan.h` |
| `StarGAN` | A GAN capable of multi-domain image-to-image translation. | `gans/star_gan.h` |

### Autoencoders

Autoencoders learn a compressed representation (encoding) of data and can be used for generative tasks, dimensionality reduction, and anomaly detection.

| Model Family | Description | Header File |
|---|---|---|
| `AE` | A standard, basic Autoencoder. | `autoencoders/ae.h` |
| `DAE` | Denoising Autoencoder, trained to reconstruct a clean image from a corrupted one. | `autoencoders/dae.h` |
| `VAE` | Variational Autoencoder, a probabilistic generative model that learns a latent space. | `autoencoders/vae.h` |
| `CAE` | Convolutional Autoencoder. | `autoencoders/cae.h` |
| `SparseAutoencoder` | An autoencoder with a sparsity penalty on the latent representation. | `autoencoders/sparse_autoencoder.h` |

### Diffusion Models

Diffusion models are a powerful new class of generative models that work by progressively adding noise to data and then learning to reverse the process.

| Model Family | Description | Header File |
|---|---|---|
| `DDPM` | Denoising Diffusion Probabilistic Models. | `diffusion/ddpm.h` |
| `DDIM` | Denoising Diffusion Implicit Models, a faster sampling variant of DDPM. | `diffusion/ddim.h` |
| `StableDiffusion`| A latent diffusion model capable of generating high-quality images from text prompts. | `diffusion/stable_diffusion.h` |
| `Imagen` | Google's text-to-image diffusion model. | `diffusion/imagen.h` |
| `GLIDE` | A text-guided diffusion model from OpenAI. | `diffusion/glide.h` |
| `DALL-E` | A multimodal model that can generate images from text. | `diffusion/dall_e.h` |

### Other Generative Architectures

| Model Family | Description | Header File |
|---|---|---|
| `PixelCNN` | An autoregressive model that generates images pixel by pixel. | `others/pixel_cnn.h` |
| `PixelRNN` | Similar to PixelCNN but uses an RNN-based architecture. | `others/pixel_rnn.h` |
| `VQ-VAE` | Vector Quantised-Variational Autoencoder, which uses a discrete latent space. | `others/vq_vae.h` |
| `Glow` | A type of normalizing flow model that uses invertible neural networks. | `others/glow.h` |
