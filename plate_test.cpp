#include <NvInfer.h>

#include <opencv2/opencv.hpp>

#include <cuda_runtime_api.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>

using namespace nvinfer1;

// ============================================================
// CONFIG
// ============================================================

static const char* RTSP_URL =
    "rtsp://admin:Radian-123@10.10.60.156:554/video/live?channel=1&subtype=0";
static const char* ENGINE_PATH =
    "model/license_plate_yolo11n.engine";

static constexpr int INPUT_W = 640;
static constexpr int INPUT_H = 640;

static constexpr float CONF_THRESHOLD = 0.25f;
static constexpr float NMS_THRESHOLD = 0.45f;

static const char* RESULTS_DIR = "results";
static constexpr int PLATE_UPSCALE = 4;
static constexpr int PLATE_PADDING = 10;


// ============================================================
// TensorRT Logger
// ============================================================

class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
            std::cout << "[TensorRT] " << msg << std::endl;
    }
};

Logger gLogger;


// ============================================================
// Detection structure
// ============================================================

struct Detection
{
    float x1;
    float y1;
    float x2;
    float y2;
    float confidence;
};


// ============================================================
// Read engine
// ============================================================

std::vector<char> readEngine(const char* path)
{
    std::ifstream file(path, std::ios::binary);

    if (!file)
    {
        throw std::runtime_error(
            std::string("Cannot open engine: ") + path
        );
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();

    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);

    file.read(data.data(), size);

    return data;
}


// ============================================================
// IoU
// ============================================================

float IoU(
    const Detection& a,
    const Detection& b)
{
    float x1 = std::max(a.x1, b.x1);
    float y1 = std::max(a.y1, b.y1);

    float x2 = std::min(a.x2, b.x2);
    float y2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);

    float intersection = w * h;

    float areaA =
        (a.x2 - a.x1) *
        (a.y2 - a.y1);

    float areaB =
        (b.x2 - b.x1) *
        (b.y2 - b.y1);

    float unionArea =
        areaA + areaB - intersection;

    if (unionArea <= 0.0f)
        return 0.0f;

    return intersection / unionArea;
}


// ============================================================
// NMS
// ============================================================

std::vector<Detection> nms(
    std::vector<Detection> detections)
{
    std::sort(
        detections.begin(),
        detections.end(),
        [](const Detection& a,
           const Detection& b)
        {
            return a.confidence > b.confidence;
        }
    );

    std::vector<Detection> result;

    std::vector<bool> suppressed(
        detections.size(),
        false
    );

    for (size_t i = 0; i < detections.size(); ++i)
    {
        if (suppressed[i])
            continue;

        result.push_back(detections[i]);

        for (size_t j = i + 1; j < detections.size(); ++j)
        {
            if (suppressed[j])
                continue;

            if (IoU(
                    detections[i],
                    detections[j])
                > NMS_THRESHOLD)
            {
                suppressed[j] = true;
            }
        }
    }

    return result;
}


// ============================================================
// Main
// ============================================================

int main()
{
    try
    {
        std::cout
            << "========================================\n"
            << " YOLO11 License Plate TensorRT Test\n"
            << "========================================\n";


        // ----------------------------------------------------
        // Load engine
        // ----------------------------------------------------

        std::cout
            << "[INFO] Loading engine...\n";

        std::vector<char> engineData =
            readEngine(ENGINE_PATH);

        std::unique_ptr<IRuntime> runtime(
            createInferRuntime(gLogger)
        );

        if (!runtime)
        {
            std::cerr
                << "[ERROR] Failed to create TensorRT runtime\n";

            return 1;
        }

        std::unique_ptr<ICudaEngine> engine(
            runtime->deserializeCudaEngine(
                engineData.data(),
                engineData.size()
            )
        );

        if (!engine)
        {
            std::cerr
                << "[ERROR] Failed to deserialize TensorRT engine\n";

            return 1;
        }

        std::cout
            << "[INFO] TensorRT engine loaded\n";


        // ----------------------------------------------------
        // Create execution context
        // ----------------------------------------------------

        std::unique_ptr<IExecutionContext> context(
            engine->createExecutionContext()
        );

        if (!context)
        {
            std::cerr
                << "[ERROR] Failed to create execution context\n";

            return 1;
        }


        // ----------------------------------------------------
        // Find tensor names
        // ----------------------------------------------------

        const char* inputName = nullptr;
        const char* outputName = nullptr;

        for (int i = 0; i < engine->getNbIOTensors(); ++i)
        {
            const char* name =
                engine->getIOTensorName(i);

            TensorIOMode mode =
                engine->getTensorIOMode(name);

            if (mode == TensorIOMode::kINPUT)
                inputName = name;

            if (mode == TensorIOMode::kOUTPUT)
                outputName = name;
        }

        std::cout
            << "[INFO] Input tensor: "
            << inputName << "\n";

        std::cout
            << "[INFO] Output tensor: "
            << outputName << "\n";


        // ----------------------------------------------------
        // Set input shape
        // ----------------------------------------------------

        context->setInputShape(
            inputName,
            Dims4(1, 3, INPUT_H, INPUT_W)
        );


        // ----------------------------------------------------
        // Get output shape
        // ----------------------------------------------------

        Dims outputDims =
            context->getTensorShape(outputName);

        std::cout
            << "[INFO] Output shape: [";

        for (int i = 0; i < outputDims.nbDims; ++i)
        {
            if (i)
                std::cout << ", ";

            std::cout
                << outputDims.d[i];
        }

        std::cout
            << "]\n";


        // ----------------------------------------------------
        // Calculate tensor sizes
        // ----------------------------------------------------

        size_t inputElements =
            1ULL *
            3 *
            INPUT_H *
            INPUT_W;

        size_t outputElements = 1;

        for (int i = 0;
             i < outputDims.nbDims;
             ++i)
        {
            outputElements *=
                outputDims.d[i];
        }


        // ----------------------------------------------------
        // Allocate GPU memory
        // ----------------------------------------------------

        void* dInput = nullptr;
        void* dOutput = nullptr;

        cudaMalloc(
            &dInput,
            inputElements * sizeof(float)
        );

        cudaMalloc(
            &dOutput,
            outputElements * sizeof(float)
        );


        // ----------------------------------------------------
        // Set tensor addresses
        // ----------------------------------------------------

        context->setTensorAddress(
            inputName,
            dInput
        );

        context->setTensorAddress(
            outputName,
            dOutput
        );


        // ----------------------------------------------------
        // CUDA stream
        // ----------------------------------------------------

        cudaStream_t stream;

        cudaStreamCreate(&stream);


        // ----------------------------------------------------
        // Open RTSP
        // ----------------------------------------------------

        std::cout
            << "[INFO] Opening RTSP stream...\n";

        cv::VideoCapture cap(
            RTSP_URL,
            cv::CAP_FFMPEG
        );

        if (!cap.isOpened())
        {
            std::cerr
                << "[ERROR] Failed to open RTSP stream\n";

            cudaFree(dInput);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);

            return 1;
        }

        std::cout
            << "[INFO] RTSP stream connected\n";


        // ----------------------------------------------------
        // FPS
        // ----------------------------------------------------

        int frameCount = 0;

        auto fpsStart =
            std::chrono::steady_clock::now();

        float fps = 0.0f;

        std::filesystem::create_directories(RESULTS_DIR);
        int plateCaptureCount = 0;


        // ----------------------------------------------------
        // Main loop
        // ----------------------------------------------------

        while (true)
        {
            cv::Mat frame;

            if (!cap.read(frame))
            {
                std::cerr
                    << "[WARNING] Failed to read frame\n";

                break;
            }


            // ------------------------------------------------
            // Original resolution
            // ------------------------------------------------

            int originalW = frame.cols;
            int originalH = frame.rows;


            // ------------------------------------------------
            // Letterbox
            // ------------------------------------------------

            float scale =
                std::min(
                    static_cast<float>(INPUT_W) /
                        originalW,
                    static_cast<float>(INPUT_H) /
                        originalH
                );

            int resizedW =
                static_cast<int>(
                    std::round(originalW * scale)
                );

            int resizedH =
                static_cast<int>(
                    std::round(originalH * scale)
                );

            cv::Mat resized;

            cv::resize(
                frame,
                resized,
                cv::Size(resizedW, resizedH)
            );

            cv::Mat inputImage(
                INPUT_H,
                INPUT_W,
                CV_8UC3,
                cv::Scalar(114, 114, 114)
            );

            int padX =
                (INPUT_W - resizedW) / 2;

            int padY =
                (INPUT_H - resizedH) / 2;

            resized.copyTo(
                inputImage(
                    cv::Rect(
                        padX,
                        padY,
                        resizedW,
                        resizedH
                    )
                )
            );


            // ------------------------------------------------
            // BGR -> RGB
            // ------------------------------------------------

            cv::cvtColor(
                inputImage,
                inputImage,
                cv::COLOR_BGR2RGB
            );


            // ------------------------------------------------
            // Convert to CHW float
            // ------------------------------------------------

            std::vector<float> inputData(
                inputElements
            );

            for (int c = 0; c < 3; ++c)
            {
                for (int y = 0; y < INPUT_H; ++y)
                {
                    for (int x = 0; x < INPUT_W; ++x)
                    {
                        inputData[
                            c * INPUT_H * INPUT_W +
                            y * INPUT_W +
                            x
                        ] =
                            inputImage.at<cv::Vec3b>(
                                y,
                                x
                            )[c] / 255.0f;
                    }
                }
            }


            // ------------------------------------------------
            // Copy input to GPU
            // ------------------------------------------------

            cudaMemcpyAsync(
                dInput,
                inputData.data(),
                inputElements * sizeof(float),
                cudaMemcpyHostToDevice,
                stream
            );


            // ------------------------------------------------
            // TensorRT inference
            // ------------------------------------------------

            if (!context->enqueueV3(stream))
            {
                std::cerr
                    << "[ERROR] TensorRT inference failed\n";

                break;
            }


            // ------------------------------------------------
            // Copy output
            // ------------------------------------------------

            std::vector<float> outputData(
                outputElements
            );

            cudaMemcpyAsync(
                outputData.data(),
                dOutput,
                outputElements * sizeof(float),
                cudaMemcpyDeviceToHost,
                stream
            );

            cudaStreamSynchronize(stream);


            // ------------------------------------------------
            // YOLO11 output
            //
            // Expected:
            // [1, 5, 8400]
            //
            // x center
            // y center
            // width
            // height
            // confidence
            // ------------------------------------------------

            std::vector<Detection> detections;

            int numCandidates =
                outputDims.d[2];

            for (int i = 0;
                 i < numCandidates;
                 ++i)
            {
                float cx =
                    outputData[i];

                float cy =
                    outputData[numCandidates + i];

                float w =
                    outputData[
                        2 * numCandidates + i
                    ];

                float h =
                    outputData[
                        3 * numCandidates + i
                    ];

                float confidence =
                    outputData[
                        4 * numCandidates + i
                    ];


                if (confidence < CONF_THRESHOLD)
                    continue;


                // --------------------------------------------
                // Remove letterbox
                // --------------------------------------------

                float x1 =
                    (cx - w / 2.0f - padX)
                    / scale;

                float y1 =
                    (cy - h / 2.0f - padY)
                    / scale;

                float x2 =
                    (cx + w / 2.0f - padX)
                    / scale;

                float y2 =
                    (cy + h / 2.0f - padY)
                    / scale;


                // --------------------------------------------
                // Clamp
                // --------------------------------------------

                x1 =
                    std::max(
                        0.0f,
                        std::min(
                            static_cast<float>(originalW - 1),
                            x1
                        )
                    );

                y1 =
                    std::max(
                        0.0f,
                        std::min(
                            static_cast<float>(originalH - 1),
                            y1
                        )
                    );

                x2 =
                    std::max(
                        0.0f,
                        std::min(
                            static_cast<float>(originalW - 1),
                            x2
                        )
                    );

                y2 =
                    std::max(
                        0.0f,
                        std::min(
                            static_cast<float>(originalH - 1),
                            y2
                        )
                    );


                if (x2 <= x1 || y2 <= y1)
                    continue;


                detections.push_back({
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence
                });
            }


            // ------------------------------------------------
            // NMS
            // ------------------------------------------------

            detections =
                nms(detections);


            // ------------------------------------------------
            // Draw detections
            // ------------------------------------------------

            // ------------------------------------------------
            if (!detections.empty())
            {
                plateCaptureCount++;

                const auto& best = detections.front();

                int x1 = std::max(
                    0,
                    static_cast<int>(best.x1) - PLATE_PADDING
                );

                int y1 = std::max(
                    0,
                    static_cast<int>(best.y1) - PLATE_PADDING
                );

                int x2 = std::min(
                    originalW,
                    static_cast<int>(best.x2) + PLATE_PADDING
                );

                int y2 = std::min(
                    originalH,
                    static_cast<int>(best.y2) + PLATE_PADDING
                );

                if (x2 > x1 && y2 > y1)
                {
                    cv::Mat plateOriginal =
                        frame(cv::Rect(
                            x1,
                            y1,
                            x2 - x1,
                            y2 - y1
                        )).clone();

                    // --------------------------------------------
                    // 1. Original plate crop
                    // --------------------------------------------

                    cv::imwrite(
                        std::string(RESULTS_DIR) + "/plate_" + std::to_string(plateCaptureCount) + "_original.jpg",
                        plateOriginal,
                        {cv::IMWRITE_JPEG_QUALITY, 100}
                    );

                    // --------------------------------------------
                    // 2. Upscale
                    // --------------------------------------------

                    cv::Mat plateUpscaled;

                    cv::resize(
                        plateOriginal,
                        plateUpscaled,
                        cv::Size(
                            plateOriginal.cols * PLATE_UPSCALE,
                            plateOriginal.rows * PLATE_UPSCALE
                        ),
                        0,
                        0,
                        cv::INTER_LANCZOS4
                    );

                    cv::imwrite(
                        std::string(RESULTS_DIR) + "/plate_" + std::to_string(plateCaptureCount) + "_upscaled.jpg",
                        plateUpscaled,
                        {cv::IMWRITE_JPEG_QUALITY, 100}
                    );

                    // --------------------------------------------
                    // 3. Enhanced version
                    // --------------------------------------------

                    cv::Mat gray;

                    cv::cvtColor(
                        plateUpscaled,
                        gray,
                        cv::COLOR_BGR2GRAY
                    );

                    cv::Ptr<cv::CLAHE> clahe =
                        cv::createCLAHE(
                            2.0,
                            cv::Size(8, 8)
                        );

                    cv::Mat contrast;

                    clahe->apply(
                        gray,
                        contrast
                    );

                    cv::Mat denoised;

                    cv::bilateralFilter(
                        contrast,
                        denoised,
                        5,
                        35,
                        35
                    );

                    cv::Mat blurred;

                    cv::GaussianBlur(
                        denoised,
                        blurred,
                        cv::Size(0, 0),
                        1.2
                    );

                    cv::Mat enhanced;

                    cv::addWeighted(
                        denoised,
                        1.5,
                        blurred,
                        -0.5,
                        0,
                        enhanced
                    );

                    cv::imwrite(
                        std::string(RESULTS_DIR) + "/plate_" + std::to_string(plateCaptureCount) + "_enhanced.jpg",
                        enhanced,
                        {cv::IMWRITE_JPEG_QUALITY, 100}
                    );

                    std::cout
                        << "[INFO] Plate images saved\n"
                        << "       Original : "
                        << RESULTS_DIR
                        << "/plate_" << plateCaptureCount << "_original.jpg\n"
                        << "       Upscaled : "
                        << RESULTS_DIR
                        << "/plate_" << plateCaptureCount << "_upscaled.jpg\n"
                        << "       Enhanced : "
                        << RESULTS_DIR
                        << "/plate_" << plateCaptureCount << "_enhanced.jpg\n"
                        << "       Confidence: "
                        << best.confidence
                        << "\n"
                        << "       Original plate size: "
                        << plateOriginal.cols
                        << "x"
                        << plateOriginal.rows
                        << "\n";

                }
            }

            for (const auto& det : detections)
            {
                cv::rectangle(
                    frame,
                    cv::Point(
                        static_cast<int>(det.x1),
                        static_cast<int>(det.y1)
                    ),
                    cv::Point(
                        static_cast<int>(det.x2),
                        static_cast<int>(det.y2)
                    ),
                    cv::Scalar(0, 255, 0),
                    2
                );

                std::string label =
                    "Plate " +
                    std::to_string(
                        det.confidence
                    ).substr(0, 4);

                cv::putText(
                    frame,
                    label,
                    cv::Point(
                        static_cast<int>(det.x1),
                        std::max(
                            25,
                            static_cast<int>(det.y1) - 8
                        )
                    ),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.6,
                    cv::Scalar(0, 255, 0),
                    2
                );
            }


            // ------------------------------------------------
            // FPS
            // ------------------------------------------------

            frameCount++;

            auto now =
                std::chrono::steady_clock::now();

            double elapsed =
                std::chrono::duration<double>(
                    now - fpsStart
                ).count();

            if (elapsed >= 1.0)
            {
                fps =
                    frameCount /
                    elapsed;

                frameCount = 0;

                fpsStart = now;
            }


            std::string fpsText =
                "YOLO11 Plate | FPS: " +
                std::to_string(fps);

            cv::putText(
                frame,
                fpsText,
                cv::Point(20, 40),
                cv::FONT_HERSHEY_SIMPLEX,
                0.8,
                cv::Scalar(0, 255, 255),
                2
            );


            // ------------------------------------------------
            // Display
            // ------------------------------------------------

            cv::imshow(
                "YOLO11 License Plate Detector",
                frame
            );

            int key =
                cv::waitKey(1) & 0xFF;

            if (key == 'q')
                break;
        }


        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------

        cap.release();

        cv::destroyAllWindows();

        cudaFree(dInput);
        cudaFree(dOutput);

        cudaStreamDestroy(stream);

        std::cout
            << "[INFO] Test finished\n";
    }
    catch (const std::exception& e)
    {
        std::cerr
            << "[ERROR] "
            << e.what()
            << std::endl;

        return 1;
    }

    return 0;
}
