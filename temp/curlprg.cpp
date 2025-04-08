#include <iostream>
#include <fstream>
#include <curl/curl.h>
#include <iomanip>

using namespace std;

// Callback to write data to file
size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream) {
    ofstream* file = static_cast<ofstream*>(stream);
    size_t written = size * nmemb;
    file->write(static_cast<char*>(ptr), written);
    return written;
}

// Callback to show download progress
int progress_callback(void* ptr, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
    const int bar_width = 50;

    if (total == 0) return 0;

    float ratio = static_cast<float>(now) / static_cast<float>(total);
    int pos = static_cast<int>(ratio * bar_width);

    cout << "\r[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) cout << "=";
        else if (i == pos) cout << ">";
        else cout << " ";
    }
    cout << "] " << fixed << setprecision(1) << ratio * 100 << "%";
    cout.flush();

    return 0;
}

int main() {
    const string url = "https://speed.hetzner.de/100MB.bin"; // Example large file
    const string output_file = "downloaded_file.bin";

    CURL* curl;
    CURLcode res;
    ofstream file(output_file, ios::binary);

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();

    if (curl && file.is_open()) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
        // Progress bar
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, progress_callback);

        cout << "Downloading from: " << url << endl;

        res = curl_easy_perform(curl);

        if (res != CURLE_OK)
            cerr << "\nDownload failed: " << curl_easy_strerror(res) << endl;
        else
            cout << "\nDownload completed: " << output_file << endl;

        curl_easy_cleanup(curl);
    } else {
        cerr << "Failed to open output file or initialize CURL." << endl;
    }

    curl_global_cleanup();
    file.close();
    return 0;
}
