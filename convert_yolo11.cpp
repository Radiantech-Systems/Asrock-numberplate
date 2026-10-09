#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <fstream>
#include <iostream>
#include <memory>
#include <string>

using namespace nvinfer1;

// ============================================================
// Simple TensorRT Logger
// ============================================================

class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

Logger gLogger;


// ============================================================
// File existence check
// ============================================================

bool fileExists(const std::string& path)
{
    std::ifstream file(path);
    return file.good();
}


// ============================================================
// Main
// ============================================================

int main()
{
    const std::string onnxFile =
        "model/license_plate_yolo11n.onnx";

    const std::string engineFile =
        "model/license_plate_yolo11n.engine";

    std::cout << "========================================\n";
    std::cout << " YOLO11 License Plate TensorRT Converter\n";
    std::cout << "========================================\n";

    // --------------------------------------------------------
    // Check ONNX file
    // --------------------------------------------------------

    if (!fileExists(onnxFile))
    {
        std::cerr << "[ERROR] ONNX model not found:\n";
        std::cerr << "        " << onnxFile << "\n";
        return 1;
    }

    std::cout << "[INFO] ONNX model: " << onnxFile << "\n";
    std::cout << "[INFO] Engine file: " << engineFile << "\n";


    // --------------------------------------------------------
    // Create TensorRT builder
    // --------------------------------------------------------

    std::unique_ptr<IBuilder> builder(
        createInferBuilder(gLogger)
    );

    if (!builder)
    {
        std::cerr << "[ERROR] Failed to create TensorRT builder\n";
        return 1;
    }

    std::cout << "[INFO] TensorRT builder created\n";


    // --------------------------------------------------------
    // Create network
    // --------------------------------------------------------

    const uint32_t explicitBatch =
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH
        );

    std::unique_ptr<INetworkDefinition> network(
        builder->createNetworkV2(explicitBatch)
    );

    if (!network)
    {
        std::cerr << "[ERROR] Failed to create network\n";
        return 1;
    }


    // --------------------------------------------------------
    // Create ONNX parser
    // --------------------------------------------------------

    std::unique_ptr<nvonnxparser::IParser> parser(
        nvonnxparser::createParser(*network, gLogger)
    );

    if (!parser)
    {
        std::cerr << "[ERROR] Failed to create ONNX parser\n";
        return 1;
    }


    // --------------------------------------------------------
    // Parse ONNX
    // --------------------------------------------------------

    std::cout << "[INFO] Parsing ONNX model...\n";

    if (!parser->parseFromFile(
            onnxFile.c_str(),
            static_cast<int>(
                ILogger::Severity::kWARNING
            )))
    {
        std::cerr << "[ERROR] Failed to parse ONNX model\n";

        for (int i = 0; i < parser->getNbErrors(); ++i)
        {
            std::cerr << "  "
                      << parser->getError(i)->desc()
                      << "\n";
        }

        return 1;
    }

    std::cout << "[INFO] ONNX parsed successfully\n";


    // --------------------------------------------------------
    // Print network information
    // --------------------------------------------------------

    std::cout << "\n[INFO] Network inputs:\n";

    for (int i = 0; i < network->getNbInputs(); ++i)
    {
        ITensor* input = network->getInput(i);

        std::cout
            << "  Input " << i
            << ": " << input->getName()
            << " shape=";

        Dims dims = input->getDimensions();

        std::cout << "[";

        for (int d = 0; d < dims.nbDims; ++d)
        {
            if (d > 0)
                std::cout << ", ";

            std::cout << dims.d[d];
        }

        std::cout << "]\n";
    }


    std::cout << "\n[INFO] Network outputs:\n";

    for (int i = 0; i < network->getNbOutputs(); ++i)
    {
        ITensor* output = network->getOutput(i);

        std::cout
            << "  Output " << i
            << ": " << output->getName()
            << " shape=";

        Dims dims = output->getDimensions();

        std::cout << "[";

        for (int d = 0; d < dims.nbDims; ++d)
        {
            if (d > 0)
                std::cout << ", ";

            std::cout << dims.d[d];
        }

        std::cout << "]\n";
    }


    // --------------------------------------------------------
    // Create builder configuration
    // --------------------------------------------------------

    std::unique_ptr<IBuilderConfig> config(
        builder->createBuilderConfig()
    );

    if (!config)
    {
        std::cerr << "[ERROR] Failed to create builder config\n";
        return 1;
    }


    // --------------------------------------------------------
    // Optimization profile for dynamic YOLO11 input
    // --------------------------------------------------------

    IOptimizationProfile* profile =
        builder->createOptimizationProfile();

    if (!profile)
    {
        std::cerr << "[ERROR] Failed to create optimization profile\\n";
        return 1;
    }

    Dims4 inputDims{1, 3, 640, 640};

    profile->setDimensions(
        "images",
        OptProfileSelector::kMIN,
        inputDims
    );

    profile->setDimensions(
        "images",
        OptProfileSelector::kOPT,
        inputDims
    );

    profile->setDimensions(
        "images",
        OptProfileSelector::kMAX,
        inputDims
    );

    config->addOptimizationProfile(profile);

    std::cout << "[INFO] Optimization profile: 1x3x640x640\\n";


    // --------------------------------------------------------
    // FP16 if supported
    // --------------------------------------------------------

    if (builder->platformHasFastFp16())
    {
        std::cout << "[INFO] TensorRT FP16 supported\n";
        config->setFlag(BuilderFlag::kFP16);
    }
    else
    {
        std::cout
            << "[INFO] FP16 not available, using FP32\n";
    }


    // --------------------------------------------------------
    // Workspace memory
    // TensorRT 10.x uses memory pool limits
    // --------------------------------------------------------

    config->setMemoryPoolLimit(
        MemoryPoolType::kWORKSPACE,
        1ULL << 30
    );

    std::cout << "[INFO] Workspace limit: 1 GB\n";


    // --------------------------------------------------------
    // Build TensorRT engine
    // --------------------------------------------------------

    std::cout << "\n[INFO] Building TensorRT engine...\n";
    std::cout << "[INFO] This may take some time...\n\n";

    std::unique_ptr<IHostMemory> serializedEngine(
        builder->buildSerializedNetwork(
            *network,
            *config
        )
    );

    if (!serializedEngine)
    {
        std::cerr
            << "[ERROR] Failed to build TensorRT engine\n";

        return 1;
    }

    std::cout << "\n[INFO] TensorRT engine built successfully\n";


    // --------------------------------------------------------
    // Save engine
    // --------------------------------------------------------

    std::ofstream engineOutput(
        engineFile,
        std::ios::binary
    );

    if (!engineOutput)
    {
        std::cerr
            << "[ERROR] Could not open engine output file:\n"
            << "        " << engineFile << "\n";

        return 1;
    }

    engineOutput.write(
        static_cast<const char*>(
            serializedEngine->data()
        ),
        serializedEngine->size()
    );

    engineOutput.close();


    // --------------------------------------------------------
    // Done
    // --------------------------------------------------------

    std::cout << "\n========================================\n";
    std::cout << " Conversion completed successfully\n";
    std::cout << "========================================\n";

    std::cout
        << "Engine: "
        << engineFile
        << "\n";

    std::cout
        << "Size: "
        << serializedEngine->size()
        << " bytes\n";

    return 0;
}
