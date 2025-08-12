#include "../third_party/edflib/edflib.h"
#include <iostream>

int main() {
    // Open the EDF file
    struct edf_hdr_struct hdr;
    int hdl = edfopen_file_readonly("/home/kami/Documents/temp/aaaaaaac_s001_t000.edf", &hdr, EDFLIB_READ_ALL_ANNOTATIONS);
    if (hdl < 0) {
        std::cerr << "Error opening EDF file: " << hdl << std::endl;
        return 1;
    }

    // Get the number of channels
    int num_signals = hdr.edfsignals;
    std::cout << "Number of signals (channels): " << num_signals << std::endl;

    // Read data from each channel
    for (int signal = 0; signal < num_signals; ++signal) {
        long long n_samples = hdr.signalparam[signal].smp_in_file;
        double* data = new double[n_samples];
        int samples_read = edfread_physical_samples(hdl, signal, n_samples, data);
        if (samples_read < 0) {
            std::cerr << "Error reading samples from signal " << signal << std::endl;
        } else {
            std::cout << "Read " << samples_read << " samples from signal " << signal << std::endl;
            for (int i = 0; i < 5 && i < n_samples; ++i) {
                std::cout <<  i << ": " << data[i] << "\t";
            }
            std::cout << std::endl;
        }
        delete[] data; // Free memory
    }

    // Close the file
    edfclose_file(hdl);
    return 0;
}