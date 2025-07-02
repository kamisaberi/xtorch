#include "../third_party/edflib/edflib.h"
#include <iostream>

int main() {
    // Open the file for writing with 1 signal
    int hdl = edfopen_file_writeonly("/home/kami/Documents/temp/example.edf", EDFLIB_FILETYPE_EDFPLUS, 1);
    if (hdl < 0) {
        std::cerr << "Error opening file for writing: " << hdl << std::endl;
        return 1;
    }

    // Set data record duration to 1 second (100000 microseconds)
    if (edf_set_datarecord_duration(hdl, 100000) < 0) {
        std::cerr << "Error setting data record duration" << std::endl;
        edfclose_file(hdl);
        return 1;
    }

    // Configure signal parameters for signal 0
    if (edf_set_label(hdl, 0, "signal0") < 0 ||                  // Signal label
        edf_set_physical_maximum(hdl, 0, 100.0) < 0 ||          // Physical maximum
        edf_set_physical_minimum(hdl, 0, -100.0) < 0 ||         // Physical minimum
        edf_set_digital_maximum(hdl, 0, 32767) < 0 ||           // Digital maximum
        edf_set_digital_minimum(hdl, 0, -32768) < 0 ||          // Digital minimum
        edf_set_samplefrequency(hdl, 0, 5) < 0) {               // Sampling rate: 5 Hz
        std::cerr << "Error setting signal parameters" << std::endl;
        edfclose_file(hdl);
        return 1;
        }

    // Write sample data for signal 0 (5 samples)
    double data[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    if (edfwrite_physical_samples(hdl, data) < 0) {
        std::cerr << "Error writing samples" << std::endl;
        edfclose_file(hdl);
        return 1;
    }

    // Close the file
    edfclose_file(hdl);
    std::cout << "File written successfully" << std::endl;
    return 0;
}