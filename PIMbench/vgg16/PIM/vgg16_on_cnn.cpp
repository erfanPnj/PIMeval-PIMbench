// Test: C++ version of VGG-16 using ON-CNN Bit-Serial In-DRAM Architecture
// Copyright (c) 2024 University of Virginia & Contributors
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <getopt.h>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "util.h"
#include "utilML.h"
#include "utilFixedPoint.h"
#include <iomanip>
#include <chrono>
#include <cassert>

using namespace std;

// Params ---------------------------------------------------------------------
typedef struct Params
{
    char *dramConfigFile;
    char *imageInputFile;
    char *kernelMatrixFile;
    bool shouldVerify;
    bool moreDebugPrints;
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./vgg16_on_cnn.out [options]"
            "\n"
            "\n    -v    should verify result with CPU"
            "\n    -i    input image file (default=generates matrix with random numbers)"
            "\n    -k    input csv file containing the kernel matrices (default=generates matrices with random numbers)"
            "\n    -c    input file containing dramsim config"
            "\n    -m    enable more debug prints (default = false)"
            "\n");
}

struct Params getInputParams(int argc, char **argv)
{
    struct Params p;
    p.dramConfigFile = nullptr;
    p.imageInputFile = nullptr;
    p.kernelMatrixFile = nullptr;
    p.shouldVerify = false;
    p.moreDebugPrints = false;

    int opt;
    while ((opt = getopt(argc, argv, "c:v:i:k:m:")) >= 0)
    {
        switch (opt)
        {
        case 'h':
            usage();
            exit(0);
            break;
        case 'c':
            p.dramConfigFile = optarg;
            break;
        case 'i':
            p.imageInputFile = optarg;
            break;
        case 'k':
            p.kernelMatrixFile = optarg;
            break;
        case 'v':
            p.shouldVerify = (*optarg == 't') ? true : false;
            break;
        case 'm':
            p.moreDebugPrints = (*optarg == 't') ? true : false;
            break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
        }
    }
    return p;
}

// =====================================================================
// In-DRAM Bit-Serial Convolution using ON-CNN Logic
// Replaces performConv() in utilML.h
// =====================================================================
void performConv_ON_CNN(const std::vector<std::vector<int>> &filterMatrix,
                        const std::vector<int> &inputMatrix,
                        std::vector<int> &outputMatrix,
                        int numRequiredPIMRows,
                        int numRequiredPIMCol,
                        int inputWidth)
{
    const int numBits = 16;
    const int M = 16;
    int filterCols = filterMatrix[0].size();
    outputMatrix.assign(numRequiredPIMCol, 0);

    PimObjId ifmObj = pimAlloc(PIM_ALLOC_V1, numRequiredPIMCol, PIM_INT16);
    if (ifmObj == -1)
    {
        std::cout << "Abort: pimAlloc failed for ifmObj" << std::endl;
        return;
    }

    PimObjId filterObj = pimAllocAssociated(ifmObj, PIM_INT16);
    PimObjId destP = pimAllocAssociated(ifmObj, PIM_INT32);
    PimObjId destSum = pimAllocAssociated(ifmObj, PIM_INT32);
    PimObjId accObj = pimAllocAssociated(ifmObj, PIM_INT32);

    pimBroadcastInt(accObj, 0);

    pimCopyHostToDevice((void *)inputMatrix.data(), ifmObj);

    PimObjId tempIfm = pimAllocAssociated(ifmObj, PIM_INT16);
    pimCopyObjectToObject(ifmObj, tempIfm);

    for (int i = 0; i < numRequiredPIMRows; i++)
    {
        int filterVal = filterMatrix[i / filterCols][i % filterCols];

        pimBroadcastInt(filterObj, filterVal);

        pimCopyObjectToObject(ifmObj, tempIfm);
        int kr = i / filterCols; // kernel row
        int kc = i % filterCols; // kernel column
        int shiftAmount = (kr * inputWidth + kc) * M;

        for (int j = 0; j < shiftAmount; j++)
        {
            pimShiftElementsRight(tempIfm);
        }

        // Hardware Block 1: OSSM (contains the 4:2 CSA logic)
        pimOSSM(tempIfm, filterObj, destP, numBits);

        // Hardware Block 2: OA Tree Reduction
        // The destP output from OSSM is a vector of 16-bit partial products. We need to reduce these to a single sum using an in-memory adder tree.
        pimOATreeReduce(destP, destSum);

        // Hardware Block 3: Carry-Propagate Accumulation
        pimAdd(accObj, destSum, accObj);
    }

    outputMatrix.resize(numRequiredPIMCol);
    pimCopyDeviceToHost(accObj, (void *)outputMatrix.data());

    pimFree(ifmObj);
    pimFree(filterObj);
    pimFree(destP);
    pimFree(destSum);
    pimFree(accObj);
    pimFree(tempIfm);
}

// =====================================================================
// Convolution Layer Orchestrator matching PIMbench standards
// =====================================================================
void conv2_ON_CNN(std::vector<std::vector<std::vector<int>>> &inputMatrix,
                  std::vector<std::vector<std::vector<int>>> &kernelMatrix,
                  std::vector<std::vector<std::vector<int>>> &resultMatrix,
                  int stride,
                  int padding)
{
    PimDeviceProperties deviceProp;
    pimGetDeviceProperties(&deviceProp);

    int inputDepth = inputMatrix.size();
    int inputHeight = inputMatrix[0].size();
    int inputWidth = inputMatrix[0][0].size();
    int kernelDepth = kernelMatrix.size();
    int kernelHeight = kernelMatrix[0].size();
    int kernelWidth = kernelMatrix[0][0].size();

    int outMatRow = std::floor((inputHeight - kernelHeight) / stride) + 1;
    int outMatCol = std::floor((inputWidth - kernelWidth) / stride) + 1;
    int numOfPIMRow = kernelHeight * kernelWidth;

    resultMatrix.resize(kernelDepth, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol, 0)));
    std::vector<int> outVector;

    // --- PROFILING MODE: ON-CNN Interleaved Memory Mapping ---
    // In the hardware architecture, each processing element (PE) receives exactly 16 channels for one pixel
    int M = 16;
    int matChunk = (inputDepth < M) ? inputDepth : M;
    int numPixels = outMatRow * outMatCol;
    int tempcol = numPixels * M; // Pad columns to M=16 boundary for OATree alignment

    int totalElements = inputHeight * inputWidth * M;
    std::vector<int> baseInterleavedImage(totalElements, 0);
    std::vector<std::vector<std::vector<int>>> allDecomp(matChunk);

    // INTERLEAVING: For each PIM row, we interleave the decomposed matrices of the first M channels (or fewer if inputDepth < M) to match the ON-CNN memory layout. This ensures that each PIM core receives the correct data for processing.
    for (int h = 0; h < inputHeight; h++)
    {
        for (int w = 0; w < inputWidth; w++)
        {
            for (int c = 0; c < matChunk; c++)
            {
                int index = (h * inputWidth + w) * M + c;
                baseInterleavedImage[index] = inputMatrix[c][h][w];
            }
        }
    }
    performConv_ON_CNN(kernelMatrix[0], baseInterleavedImage, outVector, numOfPIMRow, tempcol, inputWidth);

    int totalChunks = kernelDepth * std::ceil((float)inputDepth / M);
    std::cout << " [Profiler] Hardware simulated for 1 chunk (M=16 Interleaved). Skipping remaining "
              << totalChunks - 1 << " chunks to save time." << std::endl;
}

// =====================================================================
// Main Entry: Identical pipeline flow to standard vgg16.cpp
// =====================================================================
int main(int argc, char *argv[])
{
    struct Params params = getInputParams(argc, argv);
    std::vector<std::vector<std::vector<float>>> inputMatrix_f;
    std::vector<std::vector<std::vector<float>>> kernelMatrix_f;
    std::vector<std::vector<std::vector<int>>> inputMatrix;
    std::vector<std::vector<std::vector<int>>> kernelMatrix;

    int imageHeight = 224;
    int imageWidth = 224;
    int imageDepth = 3;
    int KernelHeight = 3;
    int kernelWidth = 3;
    int kernelDepth = 64;
    int padding = 1;

    if (params.imageInputFile == nullptr)
    {
        inputMatrix.resize(imageDepth);
        for (int i = 0; i < imageDepth; i++)
        {
            getMatrix(imageHeight, imageWidth, padding, inputMatrix[i]);
        }
    }
    else
    {
#ifdef COMPILE_WITH_JPEG
        std::string outputFile = "resized_output.jpg";
        std::vector<std::vector<std::vector<int>>> inputMatrixBeforePadding;
        readJPEG(params.imageInputFile, inputMatrixBeforePadding, imageHeight, imageWidth);
        writeResizedImage(outputFile, inputMatrixBeforePadding);
        int depth = inputMatrixBeforePadding.size();
        if (depth != imageDepth)
        {
            std::cerr << "Assertion failed: depth (" << depth << ") != imageDepth (" << imageDepth << ")\n";
            assert(depth == imageDepth && "Given input image depth does not match with expected image depth");
        }
        inputMatrix.resize(depth);
        for (int d = 0; d < depth; ++d)
        {
            addPadding(imageHeight, imageWidth, padding, inputMatrixBeforePadding[d], inputMatrix[d]);
        }
#endif
    }

    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(kernelDepth);
        for (auto &mat : kernelMatrix)
        {
            getMatrix(KernelHeight, kernelWidth, 0, mat);
        }
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.0.weight");
        if (params.shouldVerify == true)
        {
            kernelMatrix = floatToFixed(kernelMatrix_f);
        }
        else
        {
            kernelMatrix = binarizeMatrix(kernelMatrix_f);
        }
    }

    // Target Bit-Serial AP device (DRAM-AP)
    if (!createDevice(params.dramConfigFile))
        return 1;

    std::vector<std::vector<std::vector<int>>> resultMatrix1;
    std::vector<std::vector<std::vector<int>>> resultMatrix2;

    // conv1-1
    std::cout << "........starting conv1-1 (ON-CNN)........\n";
    if (params.moreDebugPrints == true)
    {
        std::cout << "Input matrix dimensions after padding: ";
        printMatrixDimensions(inputMatrix);
        std::cout << "Kernel matrix dimensions: ";
        printMatrixDimensions(kernelMatrix);
    }
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv1-1........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv1-2
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(64);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.2.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(resultMatrix1.size());
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(224, 224, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv1-2 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv1-2........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // pimShowStats();
    // return 0;

    // pool
    std::cout << "........starting pooling........\n";
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);
    std::cout << "........ending pooling........\n";

    // conv2-1
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(128);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.5.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(resultMatrix2.size());
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
    {
        addPadding(112, 112, 1, resultMatrix2[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv2-1 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv2-1........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv2-2
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(128);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.7.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(128);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(112, 112, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv2-2 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv2-2........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // pool
    resultMatrix2.clear();
    std::cout << "........starting pooling........\n";
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);
    std::cout << "........ending pooling........\n";

    // conv3-1
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(256);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.10.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(128);
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
    {
        addPadding(56, 56, 1, resultMatrix2[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv3-1 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv3-1........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv3-2
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(256);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.12.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(256);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv3-2 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv3-2........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv3-3
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(256);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.14.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(256);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv3-3 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv3-3........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // pool
    resultMatrix2.clear();
    std::cout << "........starting pooling........\n";
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);
    std::cout << "........ending pooling........\n";

    // conv4-1
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(512);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.17.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(256);
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
    {
        addPadding(28, 28, 1, resultMatrix2[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv4-1 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv4-1........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv4-2
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(512);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.19.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv4-2 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv4-2........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv4-3
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(512);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.21.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv4-3 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv4-3........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // pool
    resultMatrix2.clear();
    std::cout << "........starting pooling........\n";
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);
    std::cout << "........ending pooling........\n";

    // conv5-1
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(512);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.24.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
    {
        addPadding(14, 14, 1, resultMatrix2[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv5-1 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv5-1........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv5-2
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(512);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.26.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv5-2 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv5-2........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // conv5-3
    kernelMatrix.clear();
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(512);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.28.weight");
        kernelMatrix = (params.shouldVerify == true) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
    {
        addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
    }
    resultMatrix1.clear();
    resultMatrix1.shrink_to_fit();
    std::cout << "........starting conv5-3 (ON-CNN)........\n";
    conv2_ON_CNN(inputMatrix, kernelMatrix, resultMatrix1, 1, 1);
    std::cout << "........ending conv5-3........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // pool
    resultMatrix2.clear();
    std::cout << "........starting pooling........\n";
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);
    std::cout << "........ending pooling........\n";

    // dense layer 1
    std::vector<int> flattenedMat;
    flatten3DMat(resultMatrix2, flattenedMat);
    std::vector<std::vector<float>> denseWeight_f;
    std::vector<std::vector<int>> denseWeight;
    std::vector<int> denseOutput1;
    if (params.kernelMatrixFile == nullptr)
    {
        getMatrix(25088, 4096, 0, denseWeight);
    }
    else
    {
        denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.0.weight");
        denseWeight = (params.shouldVerify == true) ? floatToFixed(denseWeight_f) : binarizeMatrix(denseWeight_f);
    }
    std::cout << "........starting dense1........\n";
    gemv(4096, 25088, flattenedMat, denseWeight, denseOutput1);
    std::cout << "........ending dense1........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    performRelu(denseOutput1);
    std::cout << "........ending RELU........\n";

    // dense layer 2
    denseWeight.clear();
    std::vector<int> denseOutput2;
    if (params.kernelMatrixFile == nullptr)
    {
        getMatrix(4096, 4096, 0, denseWeight);
    }
    else
    {
        denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.3.weight");
        denseWeight = (params.shouldVerify == true) ? floatToFixed(denseWeight_f) : binarizeMatrix(denseWeight_f);
    }
    std::cout << "........starting dense2........\n";
    gemv(4096, 4096, denseOutput1, denseWeight, denseOutput2);
    std::cout << "........ending dense2........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    performRelu(denseOutput2);
    std::cout << "........ending RELU........\n";

    // dense layer 3
    denseWeight.clear();
    std::vector<int> denseOutput3;
    if (params.kernelMatrixFile == nullptr)
    {
        getMatrix(4096, 1000, 0, denseWeight);
    }
    else
    {
        denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.6.weight");
        denseWeight = (params.shouldVerify == true) ? floatToFixed(denseWeight_f) : binarizeMatrix(denseWeight_f);
    }
    std::cout << "........starting dense3........\n";
    gemv(1000, 4096, denseOutput2, denseWeight, denseOutput3);
    std::cout << "........ending dense3........\n";

    // Softmax
    std::vector<int> resultVector;
    std::vector<float> resultVector_f;
    softMaxPIM(denseOutput3, resultVector);
    resultVector_f = fixedToFloat(resultVector);
    std::cout << "Dimensions of the softmax output: " << resultVector.size() << std::endl;

    std::vector<std::pair<double, int>> valueIndexPairs;
    for (uint64_t i = 0; i < resultVector.size(); ++i)
    {
        valueIndexPairs.push_back(std::make_pair(resultVector[i], i));
    }
    std::sort(valueIndexPairs.begin(), valueIndexPairs.end(), std::greater<std::pair<double, int>>());

    std::cout << "Top 5 values and corresponding indices:\n";
    for (int i = 0; i < 5; ++i)
    {
        std::cout << "Value: " << valueIndexPairs[i].first << " Index: " << valueIndexPairs[i].second << std::endl;
    }

    pimShowStats();
    cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

    return 0;
}