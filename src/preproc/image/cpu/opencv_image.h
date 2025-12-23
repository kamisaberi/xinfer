#pragma once
#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/image/types.h>

namespace xinfer::preproc {

    class OpenCVImagePreprocessor : public IImagePreprocessor {
    public:
        void init(const ImagePreprocConfig& config) override;
        void process(const ImageFrame& src, core::Tensor& dst) override;

    private:
        ImagePreprocConfig m_config;
    };

}