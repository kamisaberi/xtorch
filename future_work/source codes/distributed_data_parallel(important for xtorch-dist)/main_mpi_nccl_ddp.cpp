#include "my_from_scratch_ddp_with_mpi_nccl.h" // Our new DDP module
#include "your_trainer.h"                      // Your Trainer
#include "your_model.h"                        // Your xt::Module model
#include "your_sharded_dataloader.h"           // DataLoader giving sharded data

#include <mpi.h>

// Model and DataLoader definitions (like SimpleNet and ShardedDummyDataLoader from previous examples)
// namespace xt { namespace models { struct SimpleNet : public xt::Module { ... }; }}
// struct ShardedDummyDataLoader { ... };

int main(int argc, char** argv) {
    // 1. Initialize MPI (MUST be the first thing)
    MPI_Init(&argc, &argv);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // 2. Determine Device for this Rank
    torch::Device device(torch::kCPU); // Default
    if (torch::cuda::is_available()) {
        int num_gpus = torch::cuda::device_count();
        if (num_gpus > 0) {
            // This simple assignment assumes one process per GPU.
            // More sophisticated mapping might be needed for multi-node or specific GPU assignments.
            if (rank < num_gpus) { // Only assign if enough GPUs
                 // cudaSetDevice is handled inside DDP constructor's initialize_distributed_env
                device = torch::Device(torch::kCUDA, rank); // Target device
            } else {
                if (rank==0) std::cerr << "Warning: Not enough GPUs for all MPI ranks. Rank " << rank << " will use CPU. NCCL might fail." << std::endl;
                // This rank will likely fail if NCCL is strictly used and it's on CPU, unless it's rank 0 and world_size is 1
            }
        } else {
             if (rank==0) std::cout << "No CUDA GPUs found. All ranks on CPU. NCCL cannot be used." << std::endl;
        }
    } else {
        if (rank==0) std::cout << "CUDA not available. All ranks on CPU. NCCL cannot be used." << std::endl;
    }
     if (rank == 0) std::cout << "[Main] World Size: " << world_size << ". Each rank will attempt to use its assigned device." << std::endl;


    // 3. Model
    auto underlying_model = std::make_shared<xt::models::SimpleNet>(784, 128, 10); // Your xt::Module

    // 4. Wrap with MyFromScratchDDPWithMPIAndNCCL
    // The constructor will handle moving underlying_model to the correct device for this rank.
    std::shared_ptr<xt::nn::parallel::MyFromScratchDDPWithMPIAndNCCL> ddp_model;
    try {
        ddp_model = std::make_shared<xt::nn::parallel::MyFromScratchDDPWithMPIAndNCCL>(
            underlying_model,
            device, // Pass the target device for this rank
            MPI_COMM_WORLD
        );
    } catch (const std::exception& e) {
        std::cerr << "[Rank " << rank << "] Failed to initialize DDP module: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }


    // 5. Sharded DataLoader
    size_t global_batch_size = 64;
    int total_epoch_samples = 1000 * world_size; // Ensure enough samples
    ShardedDummyDataLoader train_loader(rank, world_size, global_batch_size, total_epoch_samples);

    // 6. Optimizer (on DDP model's parameters, which are the underlying model's params)
    torch::optim::SGD optimizer(ddp_model->parameters(), torch::optim::SGDOptions(0.01));

    // 7. Loss
    auto loss_fn_impl = torch::nn::NLLLoss();
    auto loss_fn = [&](const torch::Tensor& out, const torch::Tensor& tgt){ return loss_fn_impl(out, tgt); };

    // 8. Training Loop (manual, for clarity of DDP steps)
    std::cout << "[Rank " << rank << "] Starting MPI/NCCL DDP training loop..." << std::endl;
    int num_epochs = 3;
    for (int epoch = 1; epoch <= num_epochs; ++epoch) {
        ddp_model->train(); // Sets underlying model to train mode
        double epoch_loss_sum = 0.0;
        int batches_done = 0;

        if (rank == 0) std::cout << "--- Epoch " << epoch << " ---" << std::endl;
        MPI_Barrier(MPI_COMM_WORLD); // Sync before starting epoch batches

        train_loader.reset_epoch();
        for (auto& data_target_pair : train_loader) {
            optimizer.zero_grad();

            torch::Tensor data = data_target_pair.first.to(ddp_model->device()); // Ensure data is on DDP model's device
            torch::Tensor target = data_target_pair.second.to(ddp_model->device());

            std::any output_any = ddp_model->forward({data});
            torch::Tensor output = std::any_cast<torch::Tensor>(output_any);

            torch::Tensor loss = loss_fn(output, target);
            loss.backward(); // Compute local gradients

            ddp_model->synchronize_gradients(); // *** MPI/NCCL AllReduce for gradients ***

            optimizer.step(); // Optimizer step

            epoch_loss_sum += loss.item<double>();
            batches_done++;
        }
        MPI_Barrier(MPI_COMM_WORLD); // Sync after epoch batches
        if (rank == 0 && batches_done > 0) {
             std::cout << "--- Epoch " << epoch << " Summary (Rank 0) --- Avg Loss: "
                       << epoch_loss_sum / batches_done << std::endl;
        }
    }

    if (rank == 0) {
        std::cout << "MyFromScratchDDPWithMPIAndNCCL training finished." << std::endl;
        // torch::save(underlying_model, "mpi_nccl_ddp_model_rank0.pt");
    }

    // 9. Finalize MPI
    MPI_Finalize();
    return 0;
}