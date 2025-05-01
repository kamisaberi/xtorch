#pragma once

#include <torch/torch.h>
#include <torch/data/datasets/mnist.h>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>