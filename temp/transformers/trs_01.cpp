#include <iostream>
#include <string> // For std::system if you use it

// Include the all-in-one xTorch header
#include "include/utils/utils.h"

int main() {
    std::cout << "My App (using header-only path finder) is starting." << std::endl;

    if (auto paths_opt = xt::get_library_paths()) {
        const xt::XTorchPaths& paths = *paths_opt;

        std::cout << "SUCCESS! xTorch library paths (found via header):" << std::endl;
        std::cout << "  Install Prefix:      " << paths.install_prefix << std::endl;
        std::cout << "  Venv Directory:      " << paths.venv_dir << std::endl;
        std::cout << "  Python Executable:   " << paths.python_executable << std::endl;
        std::cout << "  Actual libxTorch.so: " << paths.library_path << std::endl;


        if (!paths.python_executable.empty()) {
            std::string command = "\"" + paths.python_executable.string() + "\" --version";
            std::cout << "\nRunning command: " << command << std::endl;
            std::system(command.c_str());
        }

    } else {
        std::cerr << "ERROR: Failed to get xTorch library paths via header." << std::endl;
    }

    return 0;
}