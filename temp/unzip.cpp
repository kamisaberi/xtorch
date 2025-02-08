#include <iostream>
#include <zip.h>

int main()
{
    const char* zipFileName = "/home/kami/Documents/01.zip";

    // Step 1: Initialize libzip
    zip* archive = zip_open(zipFileName, 0, NULL);

    if (!archive) {
        std::cerr << "Failed to open the zip file."
                  << std::endl;
        return 1;
    }

    // Step 2: Get the total number of files in the zip
    // archive
    int numFiles = zip_get_num_files(archive);

    // Step 3: Loop through each file and print its contents
    for (int i = 0; i < numFiles; ++i) {
        struct zip_stat fileInfo;
        zip_stat_init(&fileInfo);

        if (zip_stat_index(archive, i, 0, &fileInfo) == 0) {
            std::cout << "File Name: " << fileInfo.name
                      << std::endl;

            // Step 4: Extract and print file contents
            zip_file* file = zip_fopen_index(archive, i, 0);
            if (file) {
                char buffer[1024];
                while (
                        zip_fread(file, buffer, sizeof(buffer))
                        > 0) {
                    std::cout.write(
                            buffer, zip_fread(file, buffer,
                                              sizeof(buffer)));
                }
                zip_fclose(file);
            }
        }
    }

    // Close the zip archive
    zip_close(archive);

    return 0;
}