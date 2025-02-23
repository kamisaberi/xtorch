#include <lzma.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>

// Function to decompress a file using LZMA
void decompressFile(const std::string& inputFile, const std::string& outputFile) {
    // Open the input file
    std::ifstream inFile(inputFile, std::ios::binary);
    if (!inFile) {
        throw std::runtime_error("Failed to open input file: " + inputFile);
    }

    // Open the output file
    std::ofstream outFile(outputFile, std::ios::binary);
    if (!outFile) {
        throw std::runtime_error("Failed to open output file: " + outputFile);
    }

    // Initialize LZMA stream
    lzma_stream strm = LZMA_STREAM_INIT;
    lzma_ret ret = lzma_stream_decoder(&strm, UINT64_MAX, LZMA_CONCATENATED);
    if (ret != LZMA_OK) {
        throw std::runtime_error("Failed to initialize LZMA decoder");
    }

    // Buffers for reading and writing
    std::vector<uint8_t> inBuffer(8192); // Input buffer
    std::vector<uint8_t> outBuffer(8192); // Output buffer

    // Decompress the file
    strm.next_in = nullptr;
    strm.avail_in = 0;
    strm.next_out = outBuffer.data();
    strm.avail_out = outBuffer.size();

    while (true) {
        // Read more data if the input buffer is empty
        if (strm.avail_in == 0 && !inFile.eof()) {
            inFile.read(reinterpret_cast<char*>(inBuffer.data()), inBuffer.size());
            strm.next_in = inBuffer.data();
            strm.avail_in = inFile.gcount();
        }

        // Decompress the data
        ret = lzma_code(&strm, inFile.eof() ? LZMA_FINISH : LZMA_RUN);

        // Write decompressed data to the output file
        if (strm.avail_out == 0 || ret == LZMA_STREAM_END) {
            size_t writeSize = outBuffer.size() - strm.avail_out;
            outFile.write(reinterpret_cast<char*>(outBuffer.data()), writeSize);
            strm.next_out = outBuffer.data();
            strm.avail_out = outBuffer.size();
        }

        // Check for errors or completion
        if (ret == LZMA_STREAM_END) {
            break; // Decompression completed
        } else if (ret != LZMA_OK) {
            lzma_end(&strm);
            throw std::runtime_error("LZMA decompression failed");
        }
    }

    // Clean up
    lzma_end(&strm);
    inFile.close();
    outFile.close();
}

int main() {
    std::string inputFile = "/home/kami/Documents/temp/xnist-images-idx3-ubyte.xz"; // Replace with your input file
    std::string outputFile = "/home/kami/Documents/temp/xnist-images-idx3-ubyte"; // Replace with your output file

    try {
        decompressFile(inputFile, outputFile);
        std::cout << "File decompressed successfully to: " << outputFile << "\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}