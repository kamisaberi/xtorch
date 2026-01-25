#include "../third_party/csv2/reader.hpp"
#include <iostream>
#include <string>

int main() {
    // Create a Reader object with default settings
    csv2::Reader<csv2::delimiter<','>,
                 csv2::quote_character<'"'>,
                 csv2::first_row_is_header<true>,
                 csv2::trim_policy::trim_whitespace> csv;

    // Memory-map the CSV file
    if (csv.mmap("/home/kami/Documents/temp/foo.csv")) {
        // Retrieve the header row (optional)
        const auto header = csv.header();

        // Iterate over each row in the CSV file
        for (const auto row : csv) {
            // Iterate over each cell in the row
            for (const auto cell : row) {
                std::string value;
                // Read the cell value
                cell.read_value(value);
                std::cout << value << "\t"; // Print with tab separation
            }
            std::cout << std::endl; // Newline after each row
        }
    } else {
        std::cerr << "Failed to open foo.csv" << std::endl;
    }

    return 0;
}