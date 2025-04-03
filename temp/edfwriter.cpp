#include "../third_party/edflib/edflib.h"
#include <iostream>

int main() {
    struct edf_hdr_struct hdr = {0};  // Initialize header structure
    // Example: Set up header (this is simplified; refer to documentation)
    hdr.filetype = EDFLIB_FILETYPE_EDFPLUS;
    hdr.edfsignals = 1;  // Number of signals
    // Set signal parameters (e.g., sampling rate, physical min/max)
    // hdr.signalparam[0].smp_in_file = ... (set number of samples)
    // hdr.signalparam[0].phys_max = ... (set physical maximum)
    // etc.

    // Open file for writing
    int hdl = edfopen_file_writeonly("/home/kami/Documents/temp/example.edf", EDFLIB_FILETYPE_EDFPLUS, hdr.edfsignals);
    if (hdl < 0) {
        std::cerr << "Error opening file for writing: " << hdl << std::endl;
        return 1;
    }

    // Write sample data (example: dummy data for signal 0)
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    if (edfwrite_physical_samples(hdl, data) < 0) {
        std::cerr << "Error writing samples" << std::endl;
    }

    // Close the file
    edfclose_file(hdl);
    std::cout << "File written successfully" << std::endl;
    return 0;
}