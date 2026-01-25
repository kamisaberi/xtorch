#pragma once

#include <torch/torch.h>
// This should bring in the necessary distributed functionalities

#include <torch/torch.h> // This should bring in most necessary components if built with USE_DISTRIBUTED=ON



// For specific ProcessGroup backends if needed for type casting or options,
// but init_process_group itself is in torch::distributed
// #include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
// #include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>

#include <string>
#include <iostream>
#include <cstdlib> // For getenv
#include <chrono>  // For std::chrono

// Helper function to initialize the distributed environment
inline void init_distributed_process_group(int& rank, int& world_size, const std::string& backend = "nccl") {
    const char* local_rank_env = std::getenv("LOCAL_RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");
    const char* rank_env = std::getenv("RANK");
    const char* master_addr_env = std::getenv("MASTER_ADDR");
    const char* master_port_env = std::getenv("MASTER_PORT");

    if (local_rank_env && world_size_env && rank_env && master_addr_env && master_port_env) {
        try {
            rank = std::stoi(rank_env);
            world_size = std::stoi(world_size_env);
            int local_rank = std::stoi(local_rank_env);

            if (torch::cuda::is_available() && torch::cuda::device_count() > local_rank) {
                torch::cuda::set_device(local_rank);
                std::cout << "Rank " << rank << " (Local Rank " << local_rank << ") using CUDA device " << local_rank << std::endl;
            } else if (torch::cuda::is_available()) {
                std::cerr << "Rank " << rank << " (Local Rank " << local_rank
                          << ") requested CUDA device " << local_rank << " but only "
                          << torch::cuda::device_count() << " devices are available. Using device 0." << std::endl;
                torch::cuda::set_device(0);
            } else {
                 std::cout << "Rank " << rank << " (Local Rank " << local_rank << "): CUDA not available or no devices. Process will run on CPU." << std::endl;
            }

            std::string init_method = "env://";

            std::cout << "Initializing DDP for Rank " << rank << "/" << world_size
                      << " with MASTER_ADDR=" << master_addr_env
                      << " MASTER_PORT=" << master_port_env
                      << " using backend: " << backend << std::endl;

            // Use the public API from torch::distributed
            torch::distributed::ProcessGroup::Options options = torch::distributed::ProcessGroup::Options::create();
            options->timeout_ = std::chrono::seconds(1800); // Default timeout 30min

            if (backend == "nccl" && !torch::cuda::is_available()){
                std::cerr << "Warning: NCCL backend requested but CUDA is not available. Attempting to fall back to Gloo." << std::endl;
                const_cast<std::string&>(backend) = "gloo"; // Modify backend string
            }

            if (backend == "nccl") {
                 torch::distributed::init_process_group(
                    torch::distributed::ProcessGroupNCCL::createProcessGroupNCCL(world_size, rank, options)
                );
            } else if (backend == "gloo") {
                 torch::distributed::init_process_group(
                    torch::distributed::ProcessGroupGloo::createProcessGroupGloo(world_size, rank, options)
                );
            } else {
                throw std::runtime_error("Unsupported DDP backend: " + backend);
            }

            std::cout << "Rank " << rank << " initialized process group." << std::endl;

        } catch (const std::exception& e) {
            std::cerr << "Error during DDP initialization for Rank " << (rank_env ? rank_env : "N/A") << ": " << e.what() << std::endl;
            // Fallback to single process mode on error
            rank = 0;
            world_size = 1;
            if (torch::cuda::is_available()) torch::cuda::set_device(0);
            std::cout << "Falling back to single-process mode due to DDP initialization error." << std::endl;
        }

    } else {
        std::cout << "DDP environment variables (RANK, WORLD_SIZE, LOCAL_RANK, MASTER_ADDR, MASTER_PORT) not fully set. Running in single-process mode." << std::endl;
        rank = 0;
        world_size = 1;
        if (torch::cuda::is_available()) {
            torch::cuda::set_device(0);
             std::cout << "Single-process mode using CUDA device 0 (if available)." << std::endl;
        } else {
             std::cout << "Single-process mode using CPU." << std::endl;
        }
    }
}

// Helper to cleanup
inline void cleanup_distributed_process_group() {
    if (torch::distributed::is_initialized()) {
        torch::distributed::destroy_process_group();
        std::cout << "Destroyed process group." << std::endl;
    }
}