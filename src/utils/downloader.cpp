#include <xinfer/utils/downloader.h>
#include <xinfer/core/logging.h>

#include <curl/curl.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>

// For SHA256, you'd typically use a library like OpenSSL or a header-only alternative.
// Here we'll just have a placeholder for the logic.
static std::string calculate_sha256(const std::string& filepath) {
    // Placeholder: In a real implementation, read the file and compute SHA256 hash.
    XINFER_LOG_WARN("SHA256 checksum verification is not yet fully implemented.");
    return "placeholder_sha256_hash";
}

namespace xinfer::utils {

// --- libcurl Callbacks (C-style) ---

// Struct to hold progress data
struct ProgressData {
    CURL* curl;
    const char* filename;
};

// Callback to write downloaded data to a file
static size_t write_data(void* ptr, size_t size, size_t nmemb, FILE* stream) {
    return fwrite(ptr, size, nmemb, stream);
}

// Callback to display the progress bar
static int progress_func(void* clientp, curl_off_t dltotal, curl_off_t dlnow, curl_off_t ultotal, curl_off_t ulnow) {
    if (dltotal <= 0) return 0;

    int progress = (int)(100.0 * dlnow / dltotal);
    int bar_width = 50;
    int pos = bar_width * progress / 100;

    std::cout << "[";
    for (int i = 0; i < bar_width; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << progress << " %\r";
    std::cout.flush();

    return 0;
}

// --- Main Downloader Logic ---

bool Downloader::download(const std::string& url, const std::string& filepath, const std::string& expected_sha256) {
    // 1. Check if file already exists
    if (std::filesystem::exists(filepath)) {
        XINFER_LOG_INFO("File already exists: " + filepath);
        // Optional: Verify checksum of existing file
        // if (!expected_sha256.empty() && calculate_sha256(filepath) == expected_sha256) ...
        return true;
    }

    CURL* curl;
    FILE* fp;
    CURLcode res;

    // Ensure parent directory exists
    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);
    curl = curl_easy_init();
    if (!curl) {
        XINFER_LOG_ERROR("Failed to initialize libcurl.");
        return false;
    }

    fp = fopen(filepath.c_str(), "wb");
    if (!fp) {
        XINFER_LOG_ERROR("Failed to open file for writing: " + filepath);
        curl_easy_cleanup(curl);
        return false;
    }

    XINFER_LOG_INFO("Downloading from: " + url);

    // Set curl options
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, fp);
    curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L); // Enable progress meter
    curl_easy_setopt(curl, CURLOPT_PROGRESSFUNCTION, progress_func);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Follow redirects

    // Perform the download
    res = curl_easy_perform(curl);

    // Cleanup
    fclose(fp);
    curl_easy_cleanup(curl);
    curl_global_cleanup();

    std::cout << std::endl; // Newline after progress bar

    if (res != CURLE_OK) {
        XINFER_LOG_ERROR("Download failed: " + std::string(curl_easy_strerror(res)));
        std::filesystem::remove(filepath); // Delete partial file
        return false;
    }

    // 2. Verify Checksum (if provided)
    if (!expected_sha256.empty()) {
        XINFER_LOG_INFO("Verifying checksum...");
        std::string actual_sha256 = calculate_sha256(filepath);
        if (actual_sha256 != expected_sha256) {
            XINFER_LOG_ERROR("Checksum mismatch! Expected " + expected_sha256 + " but got " + actual_sha256);
            std::filesystem::remove(filepath);
            return false;
        }
        XINFER_LOG_SUCCESS("Checksum OK.");
    }

    XINFER_LOG_SUCCESS("Download complete: " + filepath);
    return true;
}

} // namespace xinfer::utils