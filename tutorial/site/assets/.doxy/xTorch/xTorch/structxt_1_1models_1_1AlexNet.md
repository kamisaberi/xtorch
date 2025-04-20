

# Struct xt::models::AlexNet



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**models**](namespacext_1_1models.md) **>** [**AlexNet**](structxt_1_1models_1_1AlexNet.md)








Inherits the following classes: [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)






















## Public Attributes

| Type | Name |
| ---: | :--- |
|  torch::nn::Sequential | [**fc**](#variable-fc)   = `nullptr`<br> |
|  torch::nn::Sequential | [**fc1**](#variable-fc1)   = `nullptr`<br> |
|  torch::nn::Sequential | [**fc2**](#variable-fc2)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer1**](#variable-layer1)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer2**](#variable-layer2)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer3**](#variable-layer3)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer4**](#variable-layer4)   = `nullptr`<br> |
|  torch::nn::Sequential | [**layer5**](#variable-layer5)   = `/* multi line expression */`<br> |
































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**AlexNet**](#function-alexnet-12) (int num\_classes, int in\_channels=3) <br> |
|   | [**AlexNet**](#function-alexnet-22) (int num\_classes, int in\_channels, std::vector&lt; int64\_t &gt; input\_shape) <br> |
| virtual torch::Tensor | [**forward**](#function-forward) (torch::Tensor x) override const<br> |


## Public Functions inherited from xt::models::BaseModel

See [xt::models::BaseModel](classxt_1_1models_1_1BaseModel.md)

| Type | Name |
| ---: | :--- |
|   | [**BaseModel**](classxt_1_1models_1_1BaseModel.md#function-basemodel) () <br> |
| virtual torch::Tensor | [**forward**](classxt_1_1models_1_1BaseModel.md#function-forward) (torch::Tensor input) const = 0<br> |






















































## Public Attributes Documentation




### variable fc 

```C++
torch::nn::Sequential xt::models::AlexNet::fc;
```




<hr>



### variable fc1 

```C++
torch::nn::Sequential xt::models::AlexNet::fc1;
```




<hr>



### variable fc2 

```C++
torch::nn::Sequential xt::models::AlexNet::fc2;
```




<hr>



### variable layer1 

```C++
torch::nn::Sequential xt::models::AlexNet::layer1;
```




<hr>



### variable layer2 

```C++
torch::nn::Sequential xt::models::AlexNet::layer2;
```




<hr>



### variable layer3 

```C++
torch::nn::Sequential xt::models::AlexNet::layer3;
```




<hr>



### variable layer4 

```C++
torch::nn::Sequential xt::models::AlexNet::layer4;
```




<hr>



### variable layer5 

```C++
torch::nn::Sequential xt::models::AlexNet::layer5;
```




<hr>
## Public Functions Documentation




### function AlexNet [1/2]

```C++
xt::models::AlexNet::AlexNet (
    int num_classes,
    int in_channels=3
) 
```




<hr>



### function AlexNet [2/2]

```C++
xt::models::AlexNet::AlexNet (
    int num_classes,
    int in_channels,
    std::vector< int64_t > input_shape
) 
```




<hr>



### function forward 

```C++
virtual torch::Tensor xt::models::AlexNet::forward (
    torch::Tensor x
) override const
```



Implements [*xt::models::BaseModel::forward*](classxt_1_1models_1_1BaseModel.md#function-forward)


<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/models/cnn/alexnet/alexnet.h`

