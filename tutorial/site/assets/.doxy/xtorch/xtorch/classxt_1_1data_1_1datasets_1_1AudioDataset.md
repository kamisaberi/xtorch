

# Class xt::data::datasets::AudioDataset



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**datasets**](namespacext_1_1data_1_1datasets.md) **>** [**AudioDataset**](classxt_1_1data_1_1datasets_1_1AudioDataset.md)








Inherits the following classes: torch::data::datasets::Dataset< AudioDataset >


































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**AudioDataset**](#function-audiodataset) (const std::string & audio\_dir, const std::string & label\_file) <br> |
|  torch::data::Example | [**get**](#function-get) (size\_t index) override<br> |
|  torch::optional&lt; size\_t &gt; | [**size**](#function-size) () override const<br> |




























## Public Functions Documentation




### function AudioDataset 

```C++
inline xt::data::datasets::AudioDataset::AudioDataset (
    const std::string & audio_dir,
    const std::string & label_file
) 
```




<hr>



### function get 

```C++
inline torch::data::Example xt::data::datasets::AudioDataset::get (
    size_t index
) override
```




<hr>



### function size 

```C++
inline torch::optional< size_t > xt::data::datasets::AudioDataset::size () override const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/datasets/general/audio-dataset.h`

