#pragma once

#include <string>
#include <functional>

namespace xinfer::utils {

    /**
     * @brief A utility for downloading files over HTTP/HTTPS.
     *
     * Uses libcurl for robust, cross-platform downloads.
     */
    class Downloader {
    public:
        /**
         * @brief Downloads a file from a URL to a specified path.
         *
         * Features:
         * - Skips download if file already exists.
         * - Shows a command-line progress bar.
         * - Verifies file integrity with an optional checksum.
         *
         * @param url The URL of the file to download.
         * @param filepath The destination path to save the file.
         * @param expected_sha256 Optional: The expected SHA256 checksum of the file.
         *                        If provided and the checksum fails, the download fails.
         * @return true if the file is successfully downloaded or already exists, false otherwise.
         */
        static bool download(const std::string& url,
                             const std::string& filepath,
                             const std::string& expected_sha256 = "");
    };

} // namespace xinfer::utils