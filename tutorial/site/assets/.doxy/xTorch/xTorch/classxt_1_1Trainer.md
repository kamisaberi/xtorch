

# Class xt::Trainer



[**ClassList**](annotated.md) **>** [**xt**](namespacext.md) **>** [**Trainer**](classxt_1_1Trainer.md)










































## Public Functions

| Type | Name |
| ---: | :--- |
|   | [**Trainer**](#function-trainer) () <br> |
|  [**Trainer**](classxt_1_1Trainer.md) & | [**enable\_checkpoint**](#function-enable_checkpoint) (const std::string & path, int interval) <br> |
|  void | [**fit**](#function-fit) ([**xt::models::BaseModel**](classxt_1_1models_1_1BaseModel.md) \* model, [**xt::DataLoader**](classxt_1_1DataLoader.md)&lt; Dataset &gt; & train\_loader) <br> |
|  [**Trainer**](classxt_1_1Trainer.md) & | [**set\_loss\_fn**](#function-set_loss_fn) (std::function&lt; torch::Tensor(torch::Tensor, torch::Tensor)&gt; lossFn) <br> |
|  [**Trainer**](classxt_1_1Trainer.md) & | [**set\_max\_epochs**](#function-set_max_epochs) (int maxEpochs) <br> |
|  [**Trainer**](classxt_1_1Trainer.md) & | [**set\_optimizer**](#function-set_optimizer) (torch::optim::Optimizer \* optimizer) <br> |




























## Public Functions Documentation




### function Trainer 

```C++
xt::Trainer::Trainer () 
```




<hr>



### function enable\_checkpoint 

```C++
Trainer & xt::Trainer::enable_checkpoint (
    const std::string & path,
    int interval
) 
```




<hr>



### function fit 

```C++
template<typename Dataset>
inline void xt::Trainer::fit (
    xt::models::BaseModel * model,
    xt::DataLoader < Dataset > & train_loader
) 
```




<hr>



### function set\_loss\_fn 

```C++
Trainer & xt::Trainer::set_loss_fn (
    std::function< torch::Tensor(torch::Tensor, torch::Tensor)> lossFn
) 
```




<hr>



### function set\_max\_epochs 

```C++
Trainer & xt::Trainer::set_max_epochs (
    int maxEpochs
) 
```




<hr>



### function set\_optimizer 

```C++
Trainer & xt::Trainer::set_optimizer (
    torch::optim::Optimizer * optimizer
) 
```




<hr>

------------------------------
The documentation for this class was generated from the following file `/home/kami/Documents/cpp/models/include/trainers/trainer.h`

