

# Struct xt::models::ResNet



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**models**](namespacext_1_1models.md) **>** [**ResNet**](structxt_1_1models_1_1ResNet.md)








Inherits the following classes: [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md),  [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)


































































## Public Attributes

| Type | Name |
| ---: | :--- |
|  torch::nn::AvgPool2d | [**avgpool**](#variable-avgpool)   = `nullptr`<br> |
|  torch::nn::Sequential | [**conv1**](#variable-conv1)   = `nullptr`<br> |
|  torch::nn::Linear | [**fc**](#variable-fc)   = `nullptr`<br> |
|  int | [**inplanes**](#variable-inplanes)   = `64`<br> |
|  torch::nn::Sequential | [**layer0**](#variable-layer0)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer1**](#variable-layer1)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer2**](#variable-layer2)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer3**](#variable-layer3)   = `nullptr`<br> |
|  torch::nn::MaxPool2d | [**maxpool**](#variable-maxpool)   = `nullptr`<br> |
















































































































































































































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
|   | [**ResNet**](#function-resnet-124) (vector&lt; int &gt; layers, int num\_classes=10, int in\_channels=3) <br> |
|   | [**ResNet**](#function-resnet-224) (std::vector&lt; int &gt; layers, int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
| virtual torch::Tensor | [**forward**](#function-forward-112) (torch::Tensor x) override const<br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |
|  torch::nn::Sequential | [**makeLayerFromResidualBlock**](#function-makelayerfromresidualblock-112) (int planes, int blocks, int stride=1) <br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |




















































































































































































































































































































































## Public Attributes Documentation




### variable avgpool 

```C++
torch::nn::AvgPool2d xt::models::ResNet::avgpool;
```




<hr>



### variable conv1 

```C++
torch::nn::Sequential xt::models::ResNet::conv1;
```




<hr>



### variable fc 

```C++
torch::nn::Linear xt::models::ResNet::fc;
```




<hr>



### variable inplanes 

```C++
int xt::models::ResNet::inplanes;
```




<hr>



### variable layer0 

```C++
torch::nn::Sequential xt::models::ResNet::layer0;
```




<hr>



### variable layer1 

```C++
torch::nn::Sequential xt::models::ResNet::layer1;
```




<hr>



### variable layer2 

```C++
torch::nn::Sequential xt::models::ResNet::layer2;
```




<hr>



### variable layer3 

```C++
torch::nn::Sequential xt::models::ResNet::layer3;
```




<hr>



### variable maxpool 

```C++
torch::nn::MaxPool2d xt::models::ResNet::maxpool;
```




<hr>
## Public Functions Documentation




### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function ResNet [1/24]

```C++
xt::models::ResNet::ResNet (
    vector< int > layers,
    int num_classes=10,
    int in_channels=3
) 
```




<hr>



### function ResNet [2/24]

```C++
xt::models::ResNet::ResNet (
    std::vector< int > layers,
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function forward [1/12]

```C++
virtual torch::Tensor xt::models::ResNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>



### function makeLayerFromResidualBlock [1/12]

```C++
torch::nn::Sequential xt::models::ResNet::makeLayerFromResidualBlock (
    int planes,
    int blocks,
    int stride=1
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/models/cnn/resnet/res2net.h`

