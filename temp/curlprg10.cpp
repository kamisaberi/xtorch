// single_downloader.cpp
// A single-file libcurl downloader with tqdm-style spinner and progress bar

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <curl/curl.h>
#include <vector>

using namespace std;
using namespace std::chrono;

// Spinner style
const vector<string> tqdm_spinner = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};

size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream) {
    ofstream* file = static_cast<ofstream*>(stream);
    size_t written = size * nmemb;
    file->write(static_cast<char*>(ptr), written);
    return written;
}

class SingleDownloader {
    string url, filename;
    int bar_width = 40;
    int spinner_index = 0;
    time_point<steady_clock> start;

public:
    SingleDownloader(string u, string f)
        : url(std::move(u)), filename(std::move(f)), start(steady_clock::now()) {}

    void update_progress(curl_off_t total, curl_off_t now) {
        if (total == 0) return;

        float ratio = static_cast<float>(now) / total;
        int filled = static_cast<int>(bar_width * ratio);
        float percent = ratio * 100;

        auto now_time = steady_clock::now();
        auto elapsed = duration_cast<seconds>(now_time - start).count();
        float speed = elapsed > 0 ? now / elapsed / 1024 : 0;
        float eta = speed > 0 ? (total - now) / 1024 / speed : 0;

        cout << "\r" << tqdm_spinner[spinner_index % tqdm_spinner.size()] << " [";
        for (int i = 0; i < bar_width; ++i)
            cout << (i < filled ? "━" : " ");
        cout << "] " << fixed << setprecision(1)
             << setw(5) << percent << "% "
             << setw(6) << setprecision(0) << speed << " KB/s ETA: " << setw(3) << eta << "s" << flush;

        spinner_index++;
        this_thread::sleep_for(milliseconds(80));
    }

    static int progress_cb(void* clientp, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
        SingleDownloader* self = static_cast<SingleDownloader*>(clientp);
        self->update_progress(total, now);
        return 0;
    }

    void download() {
        CURL* curl = curl_easy_init();
        ofstream file(filename, ios::binary);

        if (curl && file.is_open()) {
            curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &file);
            curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, SingleDownloader::progress_cb);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, this);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

            cout << "📥 Downloading: " << url << endl;

            CURLcode res = curl_easy_perform(curl);

            if (res == CURLE_OK) {
                cout << "\n✅ Download completed: " << filename << endl;
            } else {
                cerr << "\n❌ Error: " << curl_easy_strerror(res) << endl;
            }

            curl_easy_cleanup(curl);
        } else {
            cerr << "❌ Failed to initialize CURL or open output file." << endl;
        }

        file.close();
    }
};

int main() {
    curl_global_init(CURL_GLOBAL_ALL);

    string url = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
    string filename = "cifar-10-binary.tar.gz";

    SingleDownloader downloader(url, filename);
    downloader.download();

    curl_global_cleanup();
    return 0;
}
