#include  "../../include/data_loaders/custom_data_loader.h"

// Forward declaration of the iterator class
class DataLoaderIterator;

// Main DataLoader class
CustomDataLoader::CustomDataLoader(const std::vector<std::pair<torch::Tensor, torch::Tensor>>& dataset,
                       int batch_size,
                       bool shuffle = false,
                       int num_workers = 0)
    : dataset_(dataset),
      batch_size_(batch_size),
      shuffle_(shuffle),
      num_workers_(std::max(0, num_workers)),
      stop_workers_(false),
      current_batch_idx_(0)
{
    // Initialize indices from 0 to dataset size - 1
    indices_.resize(dataset_.size());
    for (size_t i = 0; i < indices_.size(); ++i)
    {
        indices_[i] = i;
    }
    // Shuffle indices if shuffle is enabled
    if (shuffle_)
    {
        std::random_device rd;
        std::default_random_engine engine(rd());
        std::shuffle(indices_.begin(), indices_.end(), engine);
    }
    // Calculate the number of batches
    num_batches_ = (dataset_.size() + batch_size_ - 1) / batch_size_;

    // Start worker threads
    if (num_workers_ > 0)
    {
        for (int i = 0; i < num_workers_; ++i)
        {
            workers_.emplace_back(&CustomDataLoader::worker_thread, this);
        }
    }
}

// Destructor: stop workers and join threads
CustomDataLoader::~CustomDataLoader()
{
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        stop_workers_ = true;
    }
    queue_cond_.notify_all();
    for (auto& worker : workers_)
    {
        if (worker.joinable())
        {
            worker.join();
        }
    }
}

// Methods to get iterators for iteration support
DataLoaderIterator begin();
DataLoaderIterator end();


// Worker thread function to pre-fetch batches
void CustomDataLoader::worker_thread()
{
    while (true)
    {
        size_t batch_idx;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            // Check if we should stop or if all batches are processed
            if (stop_workers_ || current_batch_idx_ >= num_batches_)
            {
                return;
            }
            batch_idx = current_batch_idx_++;
            lock.unlock();
        }

        // Construct the batch
        size_t start = batch_idx * batch_size_;
        size_t end = std::min(start + batch_size_, dataset_.size());
        std::vector<torch::Tensor> inputs_vec;
        std::vector<torch::Tensor> targets_vec;
        for (size_t i = start; i < end; ++i)
        {
            size_t idx = indices_[i];
            inputs_vec.push_back(dataset_[idx].first.clone()); // Clone to avoid memory issues
            targets_vec.push_back(dataset_[idx].second.clone());
        }
        torch::Tensor inputs = torch::stack(inputs_vec);
        torch::Tensor targets = torch::stack(targets_vec);
        Batch batch{inputs, targets};

        // Push the batch to the queue
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            batch_queue_.push(std::move(batch));
            lock.unlock();
            queue_cond_.notify_one();
        }
    }
}

// Get the next batch (used by iterator)
Batch CustomDataLoader::get_next_batch()
{
    if (num_workers_ == 0)
    {
        // Single-threaded: construct batch directly
        size_t batch_idx = current_batch_idx_++;
        if (batch_idx >= num_batches_)
        {
            return {}; // Return empty batch to signal end
        }
        size_t start = batch_idx * batch_size_;
        size_t end = std::min(start + batch_size_, dataset_.size());
        std::vector<torch::Tensor> inputs_vec;
        std::vector<torch::Tensor> targets_vec;
        for (size_t i = start; i < end; ++i)
        {
            size_t idx = indices_[i];
            inputs_vec.push_back(dataset_[idx].first);
            targets_vec.push_back(dataset_[idx].second);
        }
        return {torch::stack(inputs_vec), torch::stack(targets_vec)};
    }
    else
    {
        // Multi-threaded: fetch from queue
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cond_.wait(lock, [this]
        {
            return !batch_queue_.empty() || stop_workers_ || current_batch_idx_ >= num_batches_;
        });
        if (batch_queue_.empty())
        {
            return {}; // Return empty batch to signal end
        }
        Batch batch = std::move(batch_queue_.front());
        batch_queue_.pop();
        return batch;
    }
}


DataLoaderIterator::DataLoaderIterator(CustomDataLoader* loader, size_t batch_idx)
    : loader_(loader), batch_idx_(batch_idx)
{
    // Pre-fetch the first batch
    if (batch_idx_ < loader_->num_batches_)
    {
        current_batch_ = loader_->get_next_batch();
    }
}

Batch DataLoaderIterator::operator*() const
{
    return current_batch_;
}

DataLoaderIterator& DataLoaderIterator::operator++()
{
    if (batch_idx_ < loader_->num_batches_)
    {
        ++batch_idx_;
        if (batch_idx_ < loader_->num_batches_)
        {
            current_batch_ = loader_->get_next_batch();
        }
    }
    return *this;
}

bool DataLoaderIterator::operator!=(const DataLoaderIterator& other) const
{
    return batch_idx_ != other.batch_idx_;
}


// Implementation of begin() and end() methods
DataLoaderIterator CustomDataLoader::begin()
{
    return DataLoaderIterator(this, 0);
}

DataLoaderIterator CustomDataLoader::end()
{
    return DataLoaderIterator(this, num_batches_);
}
