// curl_downloader.cpp
// A multithreaded libcurl-based downloader with tqdm-style spinners and progress bars

#include <iostream>
#include <fstream>
#include <iomanip>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include <curl/curl.h>

using namespace std;
using namespace std::chrono;

mutex cout_mutex;

// TQDM spinner style
const vector<string> tqdm_spinner = {"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"};

// Write data callback
size_t write_data(void* ptr, size_t size, size_t nmemb, void* stream) {
    ofstream* file = static_cast<ofstream*>(stream);
    size_t written = size * nmemb;
    file->write(static_cast<char*>(ptr), written);
    return written;
}

// Downloader class per file
class Downloader {
    string url, filename, logname;
    int id;
    int bar_width = 40;
    int spinner_index = 0;
    time_point<steady_clock> start;
    ofstream log;

public:
    Downloader(string u, string f, int i)
        : url(move(u)), filename(move(f)), id(i), logname("log_" + to_string(i) + ".txt"), start(steady_clock::now()) {
        log.open(logname);
    }

    void update_progress(curl_off_t total, curl_off_t now) {
        if (total == 0) return;
        float ratio = static_cast<float>(now) / total;
        int filled = static_cast<int>(ratio * bar_width);
        float percent = ratio * 100;

        auto now_time = steady_clock::now();
        auto elapsed = duration_cast<seconds>(now_time - start).count();
        float speed = elapsed > 0 ? now / elapsed / 1024 : 0;
        float eta = speed > 0 ? (total - now) / 1024 / speed : 0;

        {
            lock_guard<mutex> lock(cout_mutex);
            cout << "\r" << id << ": " << tqdm_spinner[spinner_index % tqdm_spinner.size()] << " [";
            for (int i = 0; i < bar_width; ++i)
                cout << (i < filled ? "━" : " ");
            cout << "] " << fixed << setprecision(1)
                 << setw(5) << percent << "% "
                 << setw(6) << setprecision(0) << speed << " KB/s ETA: " << setw(3) << eta << "s" << flush;
        }
        spinner_index++;
        this_thread::sleep_for(milliseconds(80));
    }

    static int progress_cb(void* clientp, curl_off_t total, curl_off_t now, curl_off_t, curl_off_t) {
        Downloader* self = static_cast<Downloader*>(clientp);
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
            curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, Downloader::progress_cb);
            curl_easy_setopt(curl, CURLOPT_XFERINFODATA, this);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);

            auto res = curl_easy_perform(curl);

            lock_guard<mutex> lock(cout_mutex);
            if (res == CURLE_OK) {
                cout << "\n✅ File " << id << " downloaded: " << filename << endl;
                log << "Success: " << filename << endl;
            } else {
                cerr << "\n❌ Error downloading file " << id << ": " << curl_easy_strerror(res) << endl;
                log << "Error: " << curl_easy_strerror(res) << endl;
            }

            curl_easy_cleanup(curl);
        } else {
            cerr << "❌ Could not initialize CURL or open file for " << filename << endl;
        }

        file.close();
        log.close();
    }
};

int main() {
    curl_global_init(CURL_GLOBAL_ALL);

    vector<pair<string, string>> downloads = {
        {"https://speed.hetzner.de/100MB.bin", "file1.bin"},
        {"https://speed.hetzner.de/100MB.bin", "file2.bin"}
    };

    vector<thread> threads;
    for (int i = 0; i < downloads.size(); ++i) {
        threads.emplace_back([i, &downloads]() {
            Downloader d(downloads[i].first, downloads[i].second, i + 1);
            d.download();
        });
    }

    for (auto& t : threads) t.join();

    curl_global_cleanup();
    cout << "\n🎉 All downloads finished." << endl;
    return 0;
}
