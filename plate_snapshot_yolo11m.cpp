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
#include <iomanip>
#include <string>
#include <cctype>
#include <cstdint>
#include <thread>
#include <regex>
#include <set>
#include <map>
#include <utility>

using namespace nvinfer1;

// ============================================================
// CONFIG
// ============================================================

static const char* SNAPSHOT_DIR =
    "/root/video_recorder/snapshots/vehicles/";

static const char* ENGINE_PATH =
    "model/yolo11m/license-plate-finetune-v1m.engine";

static constexpr int INPUT_W = 1280;
static constexpr int INPUT_H = 1280;

static constexpr float CONF_THRESHOLD = 0.25f;
static constexpr float NMS_THRESHOLD = 0.45f;

static const char* RESULTS_DIR = "results";


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
        // ----------------------------------------------------
        // Continuously scan saved vehicle snapshots (no RTSP stream)
        // ----------------------------------------------------

        const std::filesystem::path snapshotDir(SNAPSHOT_DIR);
        const std::filesystem::path resultsDir(RESULTS_DIR);
        std::filesystem::create_directories(resultsDir);

        if (!std::filesystem::exists(snapshotDir) ||
            !std::filesystem::is_directory(snapshotDir))
        {
            std::cerr << "[ERROR] Vehicle snapshot directory not found: "
                      << snapshotDir << "\n";
            cudaFree(dInput);
            cudaFree(dOutput);
            cudaStreamDestroy(stream);
            return 1;
        }

        const std::filesystem::path jsonPath = resultsDir / "plate_coordinates.json";
        const std::filesystem::path statePath = resultsDir / "plate_processed_state.txt";
        const std::filesystem::path jsonTempPath = resultsDir / "plate_coordinates.json.tmp";

        struct ProcessedImage
        {
            unsigned long long fileSize = 0;
            long long mtimeTicks = 0;
            int width = 0;
            int height = 0;
            std::vector<Detection> detections;
        };

        // Persistent state stores each image's file signature and detections.
        // This lets the process restart without re-running YOLO on unchanged files.
        std::map<std::string, ProcessedImage> processedState;
        {
            std::ifstream stateIn(statePath);
            std::string savedPath;
            while (stateIn)
            {
                ProcessedImage record;
                std::size_t detectionCount = 0;
                if (!(stateIn >> std::quoted(savedPath)
                              >> record.fileSize
                              >> record.mtimeTicks
                              >> record.width
                              >> record.height
                              >> detectionCount))
                    break;

                bool valid = true;
                for (std::size_t i = 0; i < detectionCount; ++i)
                {
                    Detection det{};
                    if (!(stateIn >> det.x1 >> det.y1 >> det.x2 >> det.y2 >> det.confidence))
                    {
                        valid = false;
                        break;
                    }
                    record.detections.push_back(det);
                }
                if (!valid)
                {
                    std::cerr << "[WARNING] Ignoring incomplete state record in "
                              << statePath << "\n";
                    break;
                }
                processedState[savedPath] = std::move(record);
            }
        }

        auto saveState = [&]()
        {
            const auto tempStatePath = resultsDir / "plate_processed_state.txt.tmp";
            std::ofstream stateOut(tempStatePath, std::ios::trunc);
            if (!stateOut)
            {
                std::cerr << "[ERROR] Cannot write processing state: " << tempStatePath << "\n";
                return;
            }
            stateOut << std::setprecision(9);
            for (const auto& [pathString, record] : processedState)
            {
                stateOut << std::quoted(pathString) << ' '
                         << record.fileSize << ' '
                         << record.mtimeTicks << ' '
                         << record.width << ' '
                         << record.height << ' '
                         << record.detections.size();
                for (const auto& det : record.detections)
                    stateOut << ' ' << det.x1 << ' ' << det.y1 << ' '
                             << det.x2 << ' ' << det.y2 << ' ' << det.confidence;
                stateOut << '\n';
            }
            stateOut.close();
            if (!stateOut)
            {
                std::cerr << "[ERROR] Failed while writing processing state.\n";
                return;
            }
            std::error_code ec;
            std::filesystem::rename(tempStatePath, statePath, ec);
            if (ec)
            {
                // POSIX rename normally replaces the destination; this fallback
                // handles filesystems that report an existing destination.
                std::filesystem::remove(statePath, ec);
                ec.clear();
                std::filesystem::rename(tempStatePath, statePath, ec);
            }
            if (ec)
                std::cerr << "[ERROR] Could not update processing state: " << ec.message() << "\n";
        };

        auto jsonEscape = [](const std::string& value) -> std::string
        {
            std::string escaped;
            for (unsigned char c : value)
            {
                switch (c)
                {
                    case '\\': escaped += "\\\\"; break;
                    case '"':  escaped += "\\\""; break;
                    case '\n': escaped += "\\n"; break;
                    case '\r': escaped += "\\r"; break;
                    case '\t': escaped += "\\t"; break;
                    default:
                        if (c < 0x20)
                        {
                            const char* hex = "0123456789abcdef";
                            escaped += "\\u00";
                            escaped += hex[(c >> 4) & 0x0f];
                            escaped += hex[c & 0x0f];
                        }
                        else escaped += static_cast<char>(c);
                }
            }
            return escaped;
        };

        // A filename matching a registration-number pattern is considered OCR-complete.
        // Names such as numberplate-signal-jump_....jpg do not match and remain pending.
        const std::regex plateFilenamePattern(
            R"(^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{1,4}$)");
        auto isPlateNamed = [&](const std::filesystem::path& path) -> bool
        {
            std::string stem = path.stem().string();
            std::transform(stem.begin(), stem.end(), stem.begin(),
                [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
            return std::regex_match(stem, plateFilenamePattern);
        };

        auto isJpeg = [](const std::filesystem::path& path) -> bool
        {
            std::string ext = path.extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            return ext == ".jpg" || ext == ".jpeg";
        };

        auto getSignature = [](const std::filesystem::path& path,
                               unsigned long long& fileSize,
                               long long& mtimeTicks) -> bool
        {
            try
            {
                fileSize = static_cast<unsigned long long>(std::filesystem::file_size(path));
                mtimeTicks = static_cast<long long>(
                    std::filesystem::last_write_time(path).time_since_epoch().count());
                return true;
            }
            catch (const std::filesystem::filesystem_error&)
            {
                return false;
            }
        };

        auto writeCoordinatesJson = [&]()
        {
            std::ofstream jsonOut(jsonTempPath, std::ios::trunc);
            if (!jsonOut)
            {
                std::cerr << "[ERROR] Cannot write output JSON: " << jsonTempPath << "\n";
                return;
            }

            jsonOut << "[\n";
            bool firstRecord = true;
            std::size_t jsonImages = 0;
            std::size_t jsonDetections = 0;
            for (const auto& [pathString, record] : processedState)
            {
                const std::filesystem::path imagePath(pathString);
                std::error_code existsError;
                if (!std::filesystem::is_regular_file(imagePath, existsError) ||
                    existsError || isPlateNamed(imagePath) || record.detections.empty())
                    continue;

                // Keep only records whose source image still exists under its original name.
                if (!firstRecord)
                    jsonOut << ",\n";
                firstRecord = false;
                jsonOut << "  {\"image\": \"" << jsonEscape(imagePath.string())
                        << "\", \"width\": " << record.width
                        << ", \"height\": " << record.height
                        << ", \"detections\": [";
                for (std::size_t i = 0; i < record.detections.size(); ++i)
                {
                    const auto& det = record.detections[i];
                    if (i > 0) jsonOut << ", ";
                    jsonOut << "{\"x1\": " << std::fixed << std::setprecision(2) << det.x1
                            << ", \"y1\": " << det.y1
                            << ", \"x2\": " << det.x2
                            << ", \"y2\": " << det.y2
                            << ", \"confidence\": " << det.confidence << "}";
                }
                jsonOut << "]}";
                ++jsonImages;
                jsonDetections += record.detections.size();
            }
            jsonOut << "\n]\n";
            jsonOut.close();
            if (!jsonOut)
            {
                std::cerr << "[ERROR] Failed while writing coordinate JSON.\n";
                return;
            }

            std::error_code ec;
            std::filesystem::rename(jsonTempPath, jsonPath, ec);
            if (ec)
            {
                std::filesystem::remove(jsonPath, ec);
                ec.clear();
                std::filesystem::rename(jsonTempPath, jsonPath, ec);
            }
            if (ec)
                std::cerr << "[ERROR] Could not update coordinate JSON: " << ec.message() << "\n";
            else
                std::cout << "[INFO] JSON refreshed: " << jsonImages
                          << " pending image(s), " << jsonDetections
                          << " plate detection(s) -> " << jsonPath << "\n";
        };

        // Track file stability so we don't read a JPEG while another process is still writing it.
        struct ObservedFile
        {
            unsigned long long fileSize = 0;
            long long mtimeTicks = 0;
            int stableScans = 0;
        };
        std::map<std::string, ObservedFile> observedFiles;

        std::cout << "[INFO] Continuous snapshot scanner started: " << snapshotDir << "\n"
                  << "[INFO] Persistent state: " << statePath << "\n"
                  << "[INFO] Poll interval: 2 seconds\n";

        while (true)
        {
            std::set<std::string> presentPaths;
            std::vector<std::filesystem::path> imageFiles;
            std::error_code scanError;
            for (std::filesystem::directory_iterator it(snapshotDir, scanError), end;
                 !scanError && it != end; it.increment(scanError))
            {
                if (scanError)
                    break;
                std::error_code entryError;
                if (!it->is_regular_file(entryError) || entryError || !isJpeg(it->path()))
                    continue;

                const auto imagePath = it->path();
                const std::string pathString = imagePath.string();
                presentPaths.insert(pathString);

                if (isPlateNamed(imagePath))
                {
                    // OCR has already renamed this file to its extracted plate number.
                    observedFiles.erase(pathString);
                    continue;
                }

                unsigned long long currentSize = 0;
                long long currentMtime = 0;
                if (!getSignature(imagePath, currentSize, currentMtime))
                    continue;

                auto observedIt = observedFiles.find(pathString);
                if (observedIt == observedFiles.end() ||
                    observedIt->second.fileSize != currentSize ||
                    observedIt->second.mtimeTicks != currentMtime)
                {
                    observedFiles[pathString] = {currentSize, currentMtime, 1};
                    continue; // Wait for the file to remain unchanged for another scan.
                }
                if (observedIt->second.stableScans < 2)
                {
                    ++observedIt->second.stableScans;
                    continue;
                }

                auto stateIt = processedState.find(pathString);
                if (stateIt != processedState.end() &&
                    stateIt->second.fileSize == currentSize &&
                    stateIt->second.mtimeTicks == currentMtime)
                    continue; // Already processed this exact file version.

                cv::Mat frame = cv::imread(pathString, cv::IMREAD_COLOR);
                if (frame.empty())
                {
                    std::cerr << "[WARNING] Could not read image yet: " << imagePath << "\n";
                    continue;
                }

                const int originalW = frame.cols;
                const int originalH = frame.rows;
                const float scale = std::min(
                    static_cast<float>(INPUT_W) / originalW,
                    static_cast<float>(INPUT_H) / originalH);
                const int resizedW = static_cast<int>(std::round(originalW * scale));
                const int resizedH = static_cast<int>(std::round(originalH * scale));

                cv::Mat resized;
                cv::resize(frame, resized, cv::Size(resizedW, resizedH));
                cv::Mat inputImage(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
                const int padX = (INPUT_W - resizedW) / 2;
                const int padY = (INPUT_H - resizedH) / 2;
                resized.copyTo(inputImage(cv::Rect(padX, padY, resizedW, resizedH)));
                cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);

                std::vector<float> inputData(inputElements);
                for (int c = 0; c < 3; ++c)
                    for (int y = 0; y < INPUT_H; ++y)
                        for (int x = 0; x < INPUT_W; ++x)
                            inputData[c * INPUT_H * INPUT_W + y * INPUT_W + x] =
                                inputImage.at<cv::Vec3b>(y, x)[c] / 255.0f;

                cudaError_t cudaStatus = cudaMemcpyAsync(
                    dInput, inputData.data(), inputElements * sizeof(float),
                    cudaMemcpyHostToDevice, stream);
                if (cudaStatus != cudaSuccess)
                {
                    std::cerr << "[ERROR] Copying input to GPU failed for " << imagePath
                              << ": " << cudaGetErrorString(cudaStatus) << "\n";
                    continue;
                }

                if (!context->enqueueV3(stream))
                {
                    std::cerr << "[ERROR] TensorRT inference failed for: " << imagePath << "\n";
                    continue;
                }

                std::vector<float> outputData(outputElements);
                cudaStatus = cudaMemcpyAsync(
                    outputData.data(), dOutput, outputElements * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
                if (cudaStatus != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess)
                {
                    std::cerr << "[ERROR] Copying inference output failed for: " << imagePath << "\n";
                    continue;
                }

                if (outputDims.nbDims != 3 || outputDims.d[1] < 5 || outputDims.d[2] <= 0)
                {
                    std::cerr << "[ERROR] Unexpected output tensor shape; expected [1, 5, N].\n";
                    cudaFree(dInput);
                    cudaFree(dOutput);
                    cudaStreamDestroy(stream);
                    return 1;
                }

                const int numCandidates = outputDims.d[2];
                std::vector<Detection> detections;
                for (int i = 0; i < numCandidates; ++i)
                {
                    const float cx = outputData[i];
                    const float cy = outputData[numCandidates + i];
                    const float w = outputData[2 * numCandidates + i];
                    const float h = outputData[3 * numCandidates + i];
                    const float confidence = outputData[4 * numCandidates + i];
                    if (!std::isfinite(confidence) || confidence < CONF_THRESHOLD ||
                        !std::isfinite(cx) || !std::isfinite(cy) ||
                        !std::isfinite(w) || !std::isfinite(h))
                        continue;

                    float x1 = (cx - w / 2.0f - padX) / scale;
                    float y1 = (cy - h / 2.0f - padY) / scale;
                    float x2 = (cx + w / 2.0f - padX) / scale;
                    float y2 = (cy + h / 2.0f - padY) / scale;
                    x1 = std::max(0.0f, std::min(static_cast<float>(originalW - 1), x1));
                    y1 = std::max(0.0f, std::min(static_cast<float>(originalH - 1), y1));
                    x2 = std::max(0.0f, std::min(static_cast<float>(originalW - 1), x2));
                    y2 = std::max(0.0f, std::min(static_cast<float>(originalH - 1), y2));
                    if (x2 <= x1 || y2 <= y1)
                        continue;
                    detections.push_back({x1, y1, x2, y2, confidence});
                }
                detections = nms(std::move(detections));

                // Recheck signature before marking complete in case the producer replaced it.
                unsigned long long finalSize = 0;
                long long finalMtime = 0;
                if (!getSignature(imagePath, finalSize, finalMtime) ||
                    finalSize != currentSize || finalMtime != currentMtime)
                {
                    std::cout << "[INFO] Image changed during inference; will retry: "
                              << imagePath.filename().string() << "\n";
                    observedFiles.erase(pathString);
                    continue;
                }

                ProcessedImage record;
                record.fileSize = currentSize;
                record.mtimeTicks = currentMtime;
                record.width = originalW;
                record.height = originalH;
                record.detections = std::move(detections);
                processedState[pathString] = std::move(record);
                saveState();

                const auto& savedRecord = processedState[pathString];
                std::cout << "[DONE] " << imagePath.filename().string()
                          << ": " << savedRecord.detections.size() << " plate(s) detected"
                          << (savedRecord.detections.empty() ? " (no plate; marked complete)" : " (ready for OCR)")
                          << "\n";
                for (const auto& det : savedRecord.detections)
                    std::cout << "       bbox=(" << std::fixed << std::setprecision(2)
                              << det.x1 << ", " << det.y1 << ", "
                              << det.x2 << ", " << det.y2 << ") confidence="
                              << det.confidence << "\n";
            }

            if (scanError)
                std::cerr << "[WARNING] Directory scan error: " << scanError.message() << "\n";

            // Drop state for files that have been renamed/deleted, including OCR-renamed plates.
            for (auto it = processedState.begin(); it != processedState.end(); )
            {
                const std::filesystem::path savedPath(it->first);
                std::error_code existsError;
                if (!std::filesystem::is_regular_file(savedPath, existsError) || existsError ||
                    isPlateNamed(savedPath))
                    it = processedState.erase(it);
                else
                    ++it;
            }

            // Remove stability entries for files that disappeared.
            for (auto it = observedFiles.begin(); it != observedFiles.end(); )
            {
                if (presentPaths.find(it->first) == presentPaths.end())
                    it = observedFiles.erase(it);
                else
                    ++it;
            }

            saveState();
            writeCoordinatesJson();
            std::this_thread::sleep_for(std::chrono::seconds(2));
        }

        // ----------------------------------------------------
        // Cleanup
        // ----------------------------------------------------
        cudaFree(dInput);
        cudaFree(dOutput);
        cudaStreamDestroy(stream);

        std::cout << "[INFO] Snapshot processing finished\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "[ERROR] " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
