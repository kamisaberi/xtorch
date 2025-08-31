# Image Transforms

Image transforms are essential for nearly every computer vision task. They are used for **preprocessing** (e.g., resizing images to a uniform size, normalizing pixel values) and, critically, for **data augmentation** (e.g., applying random rotations, crops, and color shifts to the training data). Data augmentation is a key technique for preventing overfitting and improving model robustness.

xTorch provides a massive library of image transforms, from basic operations to advanced, state-of-the-art augmentation techniques.

All image transforms are located under the `xt::transforms::image` namespace and their headers can be found in the `<xtorch/transforms/image/>` directory.

## General Usage

The most common way to use image transforms is to chain them together into a pipeline using `xt::transforms::Compose`. This pipeline is then passed to your `Dataset` during construction, which will apply the transformations to each image as it is loaded.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // This is a standard data augmentation pipeline for training on ImageNet
    auto training_transforms = std::make_unique<xt::transforms::Compose>(
        // Resize the smaller edge to 256, maintaining aspect ratio
        std::make_shared<xt::transforms::image::Resize>(256),
        // Randomly crop a 224x224 patch
        std::make_shared<xt::transforms::image::RandomCrop>(std::vector<int64_t>{224, 224}),
        // Randomly flip the image horizontally with a 50% probability
        std::make_shared<xt::transforms::image::RandomHorizontalFlip>(/*p=*/0.5),
        // Apply some color jitter
        std::make_shared<xt::transforms::image::ColorJitter>(),
        // Normalize the image using ImageNet mean and stddev
        std::make_shared<xt::transforms::general::Normalize>(
            std::vector<float>{0.485, 0.456, 0.406},
            std::vector<float>{0.229, 0.224, 0.225}
        )
    );

    // This pipeline would be passed to a dataset
    // auto dataset = xt::datasets::ImageFolderDataset("./data", std::move(training_transforms));
}
```

!!! info "Constructor Options"
Nearly all transforms are configurable through their constructors. This includes parameters like sizes, probabilities (`p`), rotation degrees, and more. Always refer to the specific header file in `<xtorch/transforms/image/>` for a full list of available settings.

---

## Available Transforms by Category

### Geometric Transforms

These transforms alter the spatial properties of the image.

| Transform | Description |
|---|---|
| `Resize` | Resizes the input image to a given size. |
| `Scale` | An alias for `Resize`. |
| `LongestMaxSize` | Resizes the longest edge to a max size, maintaining aspect ratio. |
| `SmallestMaxSize` | Resizes the smallest edge to a max size, maintaining aspect ratio. |
| `Crop` | Crops the image at a specified location and size. |
| `CenterCrop` | Crops the central part of the image. |
| `RandomCrop` | Crops a random part of the image. |
| `RandomResizedCrop`| Crops a random part of the image and resizes it to a specific size. A common training augmentation. |
| `Flip` | Flips the image vertically, horizontally, or both. |
| `HorizontalFlip` | Flips the image horizontally. |
| `VerticalFlip` | Flips the image vertically. |
| `RandomHorizontalFlip`| Randomly flips the image horizontally with a given probability. |
| `RandomVerticalFlip`| Randomly flips the image vertically with a given probability. |
| `RandomFlip`| Randomly flips the image horizontally and/or vertically. |
| `Pad` | Pads the image on all sides with a given value. |
| `PadIfNeeded` | Pads the image to a minimum height and width. |
| `Rotation` | Rotates the image by a specified angle. |
| `RandomRotation` | Rotates the image by a random angle within a given range. |
| `Affine` | Applies a general affine transformation to the image. |
| `RandomAffine` | Applies a random affine transformation. |
| `Perspective` | Applies a perspective transformation. |
| `RandomPerspective` | Applies a random perspective transformation. |
| `ElasticTransform`| Applies an elastic deformation to the image. |
| `GridDistortion`| Applies a grid distortion effect. |
| `OpticalDistortion`| Applies an optical barrel/pincushion distortion. |

### Color & Photometric Transforms

These transforms alter the pixel values, colors, brightness, and contrast of the image.

| Transform | Description |
|---|---|
| `ColorJitter` | Randomly changes the brightness, contrast, saturation, and hue of an image. |
| `RandomBrightnessContrast`| Randomly changes the brightness and contrast. |
| `Grayscale` | Converts the image to grayscale. |
| `RandomGrayscale` | Randomly converts the image to grayscale with a given probability. |
| `Posterize` | Reduces the number of bits for each color channel. |
| `RandomPosterize` | Randomly applies posterization. |
| `Solarize` | Inverts all pixel values above a threshold. |
| `RandomSolarize` | Randomly applies solarization. |
| `Invert` | Inverts the colors of the image. |
| `RandomInvert` | Randomly inverts the colors. |
| `Equalize` | Applies histogram equalization to the image. |
| `RandomEqualize` | Randomly applies histogram equalization. |
| `CLAHE` | Applies Contrast Limited Adaptive Histogram Equalization. |
| `ChannelShuffle` | Randomly shuffles the color channels of the image. |
| `RandomGamma` | Applies random gamma correction. |
| `RandomAdjustSharpness`| Randomly adjusts the sharpness of the image. |
| `RandomAutoContrast` | Randomly applies automatic contrast adjustment. |
| `FancyPCA` | Applies PCA-based color augmentation. |

### Augmentation & Erasing Transforms

These are advanced augmentation techniques that often involve erasing or mixing parts of images.

| Transform | Description |
|---|---|
| `Cutout` | Randomly erases one or more rectangular patches from an image. |
| `CoarseDropout`| An alternative name for `Cutout`. |
| `GridDropout` | Erases a grid of patches from an image. |
| `MaskDropout` | Applies dropout to a mask. |
| `MixUp` | Creates a new image by taking a weighted linear interpolation of two images. |
| `CutMix` | Creates a new image by cutting a patch from one image and pasting it onto another. |
| `RandomMosaic` | Combines four images into a single mosaic. |
| `RandomAugment` | Automatically applies a sequence of randomly selected augmentations (similar to AutoAugment). |
| `GridShuffle` | Shuffles patches of the image arranged in a grid. |

### Blur & Noise Transforms

| Transform | Description |
|---|---|
| `Blur` | Blurs the image using a normalized box filter. |
| `GaussianBlur` | Blurs the image using a Gaussian filter. |
| `MedianBlur` | Blurs the image using a median filter. |
| `MotionBlur` | Applies motion blur to the image. |

| `GlassBlur` | Applies a glass-like blur effect. |
| `ZoomBlur` | Applies a blur that simulates zooming. |
| `GaussianNoise`| Adds Gaussian noise to the image. |
| `NoiseInjection`| Injects random noise into the image. |

### Stylistic & Filter-Based Transforms

| Transform | Description |
|---|---|
| `Sharpen` | Sharpens the image. |
| `Emboss` | Applies an embossing filter to the image. |
| `ToSepia` | Applies a sepia filter to the image. |
| `BlackWhite` | Converts the image to black and white. |
| `Spatter` | Adds a "spatter" effect to the image, like drops on a camera lens. |
