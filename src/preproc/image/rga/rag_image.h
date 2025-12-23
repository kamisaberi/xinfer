#pragma once

#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/image/types.h>

namespace xinfer::preproc {

    /**
     * @brief Rockchip RGA (Raster Graphic Acceleration) Preprocessor.
     *
     * Uses the dedicated 2D hardware engine on RK3588/RK3568/RV1126 chips.
     * Performs Resize, Crop, and Color Space Conversion (e.g., NV12 -> RGB)
     * with near-zero CPU usage.
     */
    class RgaImagePreprocessor : public IImagePreprocessor {
    public:
        RgaImagePreprocessor();
        ~RgaImagePreprocessor() override;

        /**
         * @brief Configure the RGA session.
         * Sets target dimensions, formats, and cropping regions.
         */
        void init(const ImagePreprocConfig& config) override;

        /**
         * @brief Execute hardware acceleration.
         *
         * @param src Input image (supports raw pointers or dma_buf fds).
         * @param dst Output tensor (RGA writes directly to this memory).
         */
        void process(const ImageFrame& src, core::Tensor& dst) override;

    private:
        ImagePreprocConfig m_config;

        /**
         * @brief Helper to map xInfer ImageFormat to RGA_FORMAT enum.
         * e.g., ImageFormat::NV12 -> RK_FORMAT_YCbCr_420_SP
         */
        int map_format_to_rga(ImageFormat fmt);
    };

} // namespace xinfer::preproc