# Weather Transforms

Weather transforms are a unique and powerful category of image augmentation designed to simulate realistic adverse weather and environmental conditions.

Their primary application is to create a more robust training dataset for models that need to operate in the real world, such as those used in **autonomous driving** and outdoor robotics. By training a model on images with simulated rain, fog, and snow, you can significantly improve its performance and reliability when it encounters these conditions after deployment.

All weather transforms are located under the `xt::transforms` namespace and can be found in the `<xtorch/transforms/weather/>` header directory.

## General Usage

Weather transforms are used just like any other image transform. You can add them to a `Compose` pipeline to apply weather effects to your training images. Since these are often complex and computationally intensive, they are typically applied with a certain probability using `RandomApply` or `OneOf`.

```cpp
#include <xtorch/xtorch.h>
#include <iostream>

int main() {
    // 1. Define a pipeline that can apply various weather augmentations
    auto weather_augmentation_pipeline = std::make_unique<xt::transforms::Compose>(
        // For each image, randomly choose ONE of the following weather effects to apply
        std::make_shared<xt::transforms::OneOf>(
            std::vector<std::shared_ptr<xt::Module>>{
                // Add particle-based rain to the image
                std::make_shared<xt::transforms::weather::ParticleRain>(),
                // Add patchy fog
                std::make_shared<xt::transforms::weather::PatchyFog>(),
                // Add falling snow
                std::make_shared<xt::transforms::weather::FallingSnow>(),
                // Add sun flare
                std::make_shared<xt::transforms::weather::SunFlare>()
            }
        )
    );

    // 2. Pass the pipeline to a Dataset
    // This is especially useful for datasets like Cityscapes or KITTI.
    // auto dataset = xt::datasets::Cityscapes("./data", std::move(weather_augmentation_pipeline));

    // 3. The DataLoader will now provide images with realistic, simulated weather.
    // xt::dataloaders::ExtendedDataLoader data_loader(dataset, 4);
}
```

!!! warning "Computational Cost"
Simulating realistic weather effects can be more computationally expensive than simple geometric or color transforms. This may increase the data loading time. Consider the trade-off between the complexity of the augmentation and your training speed.

---

## Available Weather Transforms

xTorch provides a diverse set of simulations for various weather and environmental conditions.

### Fog

| Transform | Description | Header File |
|---|---|---|
| `HomogeneousFog` | Adds a uniform layer of fog across the entire image. | `homogeneous_fog.h` |
| `PatchyFog` | Simulates fog with varying density across the image. | `patchy_fog.h` |
| `DepthBasedFog` | Adds fog whose density increases with estimated distance from the camera (requires depth information). | `depth_based_fog.h` |

### Rain

| Transform | Description | Header File |
|---|---|---|
| `ParticleRain` | Simulates rain by adding particle-like streaks to the image. | `particle_rain.h` |
| `StreakBasedRain`| Simulates heavy rain with more prominent streak effects. | `streak_based_rain.h` |
| `FoggyRain` | A combined effect that simulates rain occurring within a foggy environment. | `foggy_rain.h` |
| `WetGround` | Simulates the appearance of wet, reflective ground surfaces caused by rain. | `wet_ground.h` |

### Snow & Winter

| Transform | Description | Header File |
|---|---|---|
| `FallingSnow` | Simulates the effect of actively falling snow. | `falling_snow.h` |
| `AccumulatedSnow` | Simulates the appearance of snow accumulated on surfaces. | `accumulated_snow.h` |
| `Blizzard` | Simulates heavy, wind-blown snow with reduced visibility. | `blizzard.h` |

### Other Environmental Effects

| Transform | Description | Header File |
|---|---|---|
| `SunFlare` | Simulates lens flare caused by a bright light source like the sun. | `sun_flare.h` |
| `DynamicShadows` | Adds or modifies shadows in the image to simulate different times of day. | `dynamic_shadows.h` |
| `DustSandClouds` | Simulates the effect of a dust storm or sandstorm, reducing visibility and adding a color cast. | `dust_sand_clouds.h` |
| `VegetationMotion`| Simulates the motion blur of trees and plants caused by wind. | `vegetation_motion.h` |

!!! info "Constructor Options"
Many of these effects are highly customizable. You can often control parameters like the intensity, density, direction, and color of the effect. Please refer to the specific header file for a full list of available settings.
