#pragma once

#include <torch/torch.h>
#include "../exceptions/implementation.h"


namespace torch::ext::data::datasets {
   class Flickr8k : torch::data::Dataset<Flickr8k> {

    public :
       Flickr8k();
    };
   class Flickr30k : torch::data::Dataset<Flickr30k> {

    public :
       Flickr30k();
    };
}
