

# Class xt::data::datasets::BaseDataset



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**data**](namespacext_1_1data.md) **>** [**datasets**](namespacext_1_1data_1_1datasets.md) **>** [**BaseDataset**](classxt_1_1data_1_1datasets_1_1BaseDataset.md)








Inherits the following classes: torch::data::Dataset< BaseDataset >


Inherited by the following classes: [xt::data::datasets::AgNews](classxt_1_1data_1_1datasets_1_1AgNews.md),  [xt::data::datasets::AmazonReview](classxt_1_1data_1_1datasets_1_1AmazonReview.md),  [xt::data::datasets::CIFAR10](classxt_1_1data_1_1datasets_1_1CIFAR10.md),  [xt::data::datasets::CIFAR100](classxt_1_1data_1_1datasets_1_1CIFAR100.md),  [xt::data::datasets::CMUArctic](classxt_1_1data_1_1datasets_1_1CMUArctic.md),  [xt::data::datasets::COLA](classxt_1_1data_1_1datasets_1_1COLA.md),  [xt::data::datasets::CREStereo](classxt_1_1data_1_1datasets_1_1CREStereo.md),  [xt::data::datasets::CSVDataset](classxt_1_1data_1_1datasets_1_1CSVDataset.md),  [xt::data::datasets::Caltech101](classxt_1_1data_1_1datasets_1_1Caltech101.md),  [xt::data::datasets::Caltech256](classxt_1_1data_1_1datasets_1_1Caltech256.md),  [xt::data::datasets::CarlaStereo](classxt_1_1data_1_1datasets_1_1CarlaStereo.md),  [xt::data::datasets::CelebA](classxt_1_1data_1_1datasets_1_1CelebA.md),  [xt::data::datasets::Cityscapes](classxt_1_1data_1_1datasets_1_1Cityscapes.md),  [xt::data::datasets::CoNLL2000Chunking](classxt_1_1data_1_1datasets_1_1CoNLL2000Chunking.md),  [xt::data::datasets::CocoCaptions](classxt_1_1data_1_1datasets_1_1CocoCaptions.md),  [xt::data::datasets::CocoDetection](classxt_1_1data_1_1datasets_1_1CocoDetection.md),  [xt::data::datasets::CommonVoice](classxt_1_1data_1_1datasets_1_1CommonVoice.md),  [xt::data::datasets::Country211](classxt_1_1data_1_1datasets_1_1Country211.md),  [xt::data::datasets::CremaD](classxt_1_1data_1_1datasets_1_1CremaD.md),  [xt::data::datasets::DBPedia](classxt_1_1data_1_1datasets_1_1DBPedia.md),  [xt::data::datasets::DTD](classxt_1_1data_1_1datasets_1_1DTD.md),  [xt::data::datasets::EDFDataset](classxt_1_1data_1_1datasets_1_1EDFDataset.md),  [xt::data::datasets::ESC](classxt_1_1data_1_1datasets_1_1ESC.md),  [xt::data::datasets::ETH3DStereo](classxt_1_1data_1_1datasets_1_1ETH3DStereo.md),  [xt::data::datasets::EnWik](classxt_1_1data_1_1datasets_1_1EnWik.md),  [xt::data::datasets::EuroSAT](classxt_1_1data_1_1datasets_1_1EuroSAT.md),  [xt::data::datasets::FER2013](classxt_1_1data_1_1datasets_1_1FER2013.md),  [xt::data::datasets::FGVCAircraft](classxt_1_1data_1_1datasets_1_1FGVCAircraft.md),  [xt::data::datasets::FakeData](classxt_1_1data_1_1datasets_1_1FakeData.md),  [xt::data::datasets::FallingThingsStereo](classxt_1_1data_1_1datasets_1_1FallingThingsStereo.md),  [xt::data::datasets::Flickr30k](classxt_1_1data_1_1datasets_1_1Flickr30k.md),  [xt::data::datasets::Flickr8k](classxt_1_1data_1_1datasets_1_1Flickr8k.md),  [xt::data::datasets::Flowers102](classxt_1_1data_1_1datasets_1_1Flowers102.md),  [xt::data::datasets::FlyingChairs](classxt_1_1data_1_1datasets_1_1FlyingChairs.md),  [xt::data::datasets::FlyingThings3D](classxt_1_1data_1_1datasets_1_1FlyingThings3D.md),  [xt::data::datasets::Food101](classxt_1_1data_1_1datasets_1_1Food101.md),  [xt::data::datasets::GTSRB](classxt_1_1data_1_1datasets_1_1GTSRB.md),  [xt::data::datasets::GTZAN](classxt_1_1data_1_1datasets_1_1GTZAN.md),  [xt::data::datasets::HD1K](classxt_1_1data_1_1datasets_1_1HD1K.md),  [xt::data::datasets::HMDB51](classxt_1_1data_1_1datasets_1_1HMDB51.md),  [xt::data::datasets::IMDB](classxt_1_1data_1_1datasets_1_1IMDB.md),  [xt::data::datasets::INaturalist](classxt_1_1data_1_1datasets_1_1INaturalist.md),  [xt::data::datasets::IWSLT](classxt_1_1data_1_1datasets_1_1IWSLT.md),  [xt::data::datasets::ImageFolder](classxt_1_1data_1_1datasets_1_1ImageFolder.md),  [xt::data::datasets::ImageNet](classxt_1_1data_1_1datasets_1_1ImageNet.md),  [xt::data::datasets::Imagenette](classxt_1_1data_1_1datasets_1_1Imagenette.md),  [xt::data::datasets::InStereo2k](classxt_1_1data_1_1datasets_1_1InStereo2k.md),  [xt::data::datasets::Kinetics](classxt_1_1data_1_1datasets_1_1Kinetics.md),  [xt::data::datasets::Kitti](classxt_1_1data_1_1datasets_1_1Kitti.md),  [xt::data::datasets::Kitti2012Stereo](classxt_1_1data_1_1datasets_1_1Kitti2012Stereo.md),  [xt::data::datasets::Kitti2015Stereo](classxt_1_1data_1_1datasets_1_1Kitti2015Stereo.md),  [xt::data::datasets::KittiFlow](classxt_1_1data_1_1datasets_1_1KittiFlow.md),  [xt::data::datasets::LFW](classxt_1_1data_1_1datasets_1_1LFW.md),  [xt::data::datasets::LSUN](classxt_1_1data_1_1datasets_1_1LSUN.md),  [xt::data::datasets::LibriSpeech](classxt_1_1data_1_1datasets_1_1LibriSpeech.md),  [xt::data::datasets::LjSpeech](classxt_1_1data_1_1datasets_1_1LjSpeech.md),  [xt::data::datasets::MNISTBase](classxt_1_1data_1_1datasets_1_1MNISTBase.md),  [xt::data::datasets::MNLI](classxt_1_1data_1_1datasets_1_1MNLI.md),  [xt::data::datasets::MRPC](classxt_1_1data_1_1datasets_1_1MRPC.md),  [xt::data::datasets::MULTI](classxt_1_1data_1_1datasets_1_1MULTI.md),  [xt::data::datasets::Middlebury2014Stereo](classxt_1_1data_1_1datasets_1_1Middlebury2014Stereo.md),  [xt::data::datasets::MovingMNIST](classxt_1_1data_1_1datasets_1_1MovingMNIST.md),  [xt::data::datasets::Omniglot](classxt_1_1data_1_1datasets_1_1Omniglot.md),  [xt::data::datasets::OxfordIIITPet](classxt_1_1data_1_1datasets_1_1OxfordIIITPet.md),  [xt::data::datasets::PCAM](classxt_1_1data_1_1datasets_1_1PCAM.md),  [xt::data::datasets::PennTreebank](classxt_1_1data_1_1datasets_1_1PennTreebank.md),  [xt::data::datasets::PhotoTour](classxt_1_1data_1_1datasets_1_1PhotoTour.md),  [xt::data::datasets::Places365](classxt_1_1data_1_1datasets_1_1Places365.md),  [xt::data::datasets::QNLI](classxt_1_1data_1_1datasets_1_1QNLI.md),  [xt::data::datasets::QQP](classxt_1_1data_1_1datasets_1_1QQP.md),  [xt::data::datasets::RTE](classxt_1_1data_1_1datasets_1_1RTE.md),  [xt::data::datasets::RenderedSST2](classxt_1_1data_1_1datasets_1_1RenderedSST2.md),  [xt::data::datasets::SBDataset](classxt_1_1data_1_1datasets_1_1SBDataset.md),  [xt::data::datasets::SBU](classxt_1_1data_1_1datasets_1_1SBU.md),  [xt::data::datasets::SEMEION](classxt_1_1data_1_1datasets_1_1SEMEION.md),  [xt::data::datasets::SNLI](classxt_1_1data_1_1datasets_1_1SNLI.md),  [xt::data::datasets::SST](classxt_1_1data_1_1datasets_1_1SST.md),  [xt::data::datasets::STL10](classxt_1_1data_1_1datasets_1_1STL10.md),  [xt::data::datasets::STSB](classxt_1_1data_1_1datasets_1_1STSB.md),  [xt::data::datasets::SUN397](classxt_1_1data_1_1datasets_1_1SUN397.md),  [xt::data::datasets::SVHN](classxt_1_1data_1_1datasets_1_1SVHN.md),  [xt::data::datasets::SceneFlowStereo](classxt_1_1data_1_1datasets_1_1SceneFlowStereo.md),  [xt::data::datasets::Sintel](classxt_1_1data_1_1datasets_1_1Sintel.md),  [xt::data::datasets::SintelStereo](classxt_1_1data_1_1datasets_1_1SintelStereo.md),  [xt::data::datasets::SogouNews](classxt_1_1data_1_1datasets_1_1SogouNews.md),  [xt::data::datasets::SpeechCommands](classxt_1_1data_1_1datasets_1_1SpeechCommands.md),  [xt::data::datasets::StackedAudioDataset](classxt_1_1data_1_1datasets_1_1StackedAudioDataset.md),  [xt::data::datasets::StackedCSVDataset](classxt_1_1data_1_1datasets_1_1StackedCSVDataset.md),  [xt::data::datasets::StackedEDFDataset](classxt_1_1data_1_1datasets_1_1StackedEDFDataset.md),  [xt::data::datasets::StackedTextDataset](classxt_1_1data_1_1datasets_1_1StackedTextDataset.md),  [xt::data::datasets::StackedTimeSeriesDataset](classxt_1_1data_1_1datasets_1_1StackedTimeSeriesDataset.md),  [xt::data::datasets::StackedVideoDataset](classxt_1_1data_1_1datasets_1_1StackedVideoDataset.md),  [xt::data::datasets::StanfordCars](classxt_1_1data_1_1datasets_1_1StanfordCars.md),  [xt::data::datasets::TIMIT](classxt_1_1data_1_1datasets_1_1TIMIT.md),  [xt::data::datasets::Tedlium](classxt_1_1data_1_1datasets_1_1Tedlium.md),  [xt::data::datasets::TensorDataset](classxt_1_1data_1_1datasets_1_1TensorDataset.md),  [xt::data::datasets::TextDataset](classxt_1_1data_1_1datasets_1_1TextDataset.md),  [xt::data::datasets::TimeSeriesDataset](classxt_1_1data_1_1datasets_1_1TimeSeriesDataset.md),  [xt::data::datasets::UCF101](classxt_1_1data_1_1datasets_1_1UCF101.md),  [xt::data::datasets::UDPOS](classxt_1_1data_1_1datasets_1_1UDPOS.md),  [xt::data::datasets::USPS](classxt_1_1data_1_1datasets_1_1USPS.md),  [xt::data::datasets::UrbanSound](classxt_1_1data_1_1datasets_1_1UrbanSound.md),  [xt::data::datasets::VCTK](classxt_1_1data_1_1datasets_1_1VCTK.md),  [xt::data::datasets::VOCDetection](classxt_1_1data_1_1datasets_1_1VOCDetection.md),  [xt::data::datasets::VOCSegmentation](classxt_1_1data_1_1datasets_1_1VOCSegmentation.md),  [xt::data::datasets::VideoDataset](classxt_1_1data_1_1datasets_1_1VideoDataset.md),  [xt::data::datasets::VoxCeleb](classxt_1_1data_1_1datasets_1_1VoxCeleb.md),  [xt::data::datasets::WIDERFace](classxt_1_1data_1_1datasets_1_1WIDERFace.md),  [xt::data::datasets::WMT](classxt_1_1data_1_1datasets_1_1WMT.md),  [xt::data::datasets::WNLI](classxt_1_1data_1_1datasets_1_1WNLI.md),  [xt::data::datasets::WikiText](classxt_1_1data_1_1datasets_1_1WikiText.md),  [xt::data::datasets::YahooAnswers](classxt_1_1data_1_1datasets_1_1YahooAnswers.md),  [xt::data::datasets::YelpReview](classxt_1_1data_1_1datasets_1_1YelpReview.md),  [xt::data::datasets::YesNo](classxt_1_1data_1_1datasets_1_1YesNo.md)












## Public Types

| Type | Name |
| ---: | :--- |
| typedef vector&lt; std::function&lt; torch::Tensor(torch::Tensor)&gt; &gt; | [**TransformType**](#typedef-transformtype)  <br> |




## Public Attributes

| Type | Name |
| ---: | :--- |
|  [**xt::data::transforms::Compose**](classxt_1_1data_1_1transforms_1_1Compose.md) | [**compose**](#variable-compose)  <br> |
|  std::vector&lt; torch::Tensor &gt; | [**data**](#variable-data)  <br> |
|  fs::path | [**dataset\_path**](#variable-dataset_path)  <br> |
|  bool | [**download**](#variable-download)   = `false`<br> |
|  std::vector&lt; uint8\_t &gt; | [**labels**](#variable-labels)  <br> |
|  DataMode | [**mode**](#variable-mode)   = `DataMode::TRAIN`<br> |
|  fs::path | [**root**](#variable-root)  <br> |
|  vector&lt; std::function&lt; torch::Tensor(torch::Tensor)&gt; &gt; | [**transforms**](#variable-transforms)   = `{}`<br> |
















## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**BaseDataset**](#function-basedataset-14) (const std::string & root) <br> |
|   | [**BaseDataset**](#function-basedataset-24) (const std::string & root, DataMode mode) <br> |
|   | [**BaseDataset**](#function-basedataset-34) (const std::string & root, DataMode mode, bool download) <br> |
|   | [**BaseDataset**](#function-basedataset-44) (const std::string & root, DataMode mode, bool download, vector&lt; std::function&lt; torch::Tensor(torch::Tensor)&gt; &gt; transforms) <br> |
|  torch::data::Example | [**get**](#function-get) (size\_t index) override<br> |
|  torch::optional&lt; size\_t &gt; | [**size**](#function-size) () override const<br> |




























## Public Types Documentation




### typedef TransformType 

```C++
using xt::data::datasets::BaseDataset::TransformType =  vector<std::function<torch::Tensor(torch::Tensor)> >;
```




<hr>
## Public Attributes Documentation




### variable compose 

```C++
xt::data::transforms::Compose xt::data::datasets::BaseDataset::compose;
```




<hr>



### variable data 

```C++
std::vector<torch::Tensor> xt::data::datasets::BaseDataset::data;
```




<hr>



### variable dataset\_path 

```C++
fs::path xt::data::datasets::BaseDataset::dataset_path;
```




<hr>



### variable download 

```C++
bool xt::data::datasets::BaseDataset::download;
```




<hr>



### variable labels 

```C++
std::vector<uint8_t> xt::data::datasets::BaseDataset::labels;
```




<hr>



### variable mode 

```C++
DataMode xt::data::datasets::BaseDataset::mode;
```




<hr>



### variable root 

```C++
fs::path xt::data::datasets::BaseDataset::root;
```




<hr>



### variable transforms 

```C++
vector<std::function<torch::Tensor(torch::Tensor)> > xt::data::datasets::BaseDataset::transforms;
```




<hr>
## Public Functions Documentation




### function BaseDataset [1/4]

```C++
xt::data::datasets::BaseDataset::BaseDataset (
    const std::string & root
) 
```




<hr>



### function BaseDataset [2/4]

```C++
xt::data::datasets::BaseDataset::BaseDataset (
    const std::string & root,
    DataMode mode
) 
```




<hr>



### function BaseDataset [3/4]

```C++
xt::data::datasets::BaseDataset::BaseDataset (
    const std::string & root,
    DataMode mode,
    bool download
) 
```




<hr>



### function BaseDataset [4/4]

```C++
xt::data::datasets::BaseDataset::BaseDataset (
    const std::string & root,
    DataMode mode,
    bool download,
    vector< std::function< torch::Tensor(torch::Tensor)> > transforms
) 
```




<hr>



### function get 

```C++
torch::data::Example xt::data::datasets::BaseDataset::get (
    size_t index
) override
```




<hr>



### function size 

```C++
torch::optional< size_t > xt::data::datasets::BaseDataset::size () override const
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/datasets/base/base.h`

