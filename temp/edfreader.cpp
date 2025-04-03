#include "../third_party/edflib/edflib.h"
#include <iostream>

int main() {
    struct edf_hdr_struct hdr;  // Structure to hold header information
    // Open the file for reading, including annotations if EDF+
    int hdl = edfopen_file_readonly("/home/kami/Documents/temp/example.edf", &hdr, EDFLIB_READ_ALL_ANNOTATIONS);
    if (hdl < 0) {
        std::cerr << "Error opening file: " << hdl << std::endl;
        return 1;
    }

    // Get number of signals (adjust for EDF+ if necessary)
    int num_signals = hdr.edfsignals;
    std::cout << "Number of signals: " << num_signals << std::endl;

    // Read data for signal 0
    int signal = 0;
    long long n_samples = hdr.signalparam[signal].smp_in_file;
    double* data = new double[n_samples];
    int samples_read = edfread_physical_samples(hdl, signal, n_samples, data);
    if (samples_read < 0) {
        std::cerr << "Error reading samples" << std::endl;
    } else {
        std::cout << "Read " << samples_read << " samples from signal 0" << std::endl;
        // Process data (e.g., print first few samples)
        for (int i = 0; i < 5 && i < n_samples; ++i) {
            std::cout << "Sample " << i << ": " << data[i] << std::endl;
        }
    }

    // Clean up
    edfclose_file(hdl);
    delete[] data;
    return 0;
}