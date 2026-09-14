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
                        const std::vector<std::vector<int>> &inputMatrix,
                        std::vector<int> &outputMatrix,
                        int numRequiredPIMRows,
                        int numRequiredPIMCol)
{
    const int numBits = 16;
    int filterCols = filterMatrix[0].size();
    outputMatrix.assign(numRequiredPIMCol, 0);

    // 1. Allocation strictly within physical device limits (~3.2M elements fits safely in 2048 cores)
    PimObjId ifmObj = pimAlloc(PIM_ALLOC_V1, numRequiredPIMCol, PIM_INT16);
    if (ifmObj == -1)
    {
        std::cout << "Abort: pimAlloc failed for ifmObj" << std::endl;
        return;
    }

    PimObjId filterObj = pimAllocAssociated(ifmObj, PIM_INT16);
    PimObjId destP = pimAllocAssociated(ifmObj, PIM_INT32);
    PimObjId destSum = pimAllocAssociated(ifmObj, PIM_INT32); // For hardware profiling
    PimObjId destQ = pimAllocAssociated(ifmObj, PIM_INT32);
    PimObjId destQM = pimAllocAssociated(ifmObj, PIM_INT32);
    PimObjId q_sign = pimAllocAssociated(ifmObj, PIM_BOOL);
    PimObjId q_mag = pimAllocAssociated(ifmObj, PIM_BOOL);
    PimObjId accObj = pimAllocAssociated(ifmObj, PIM_INT32);

    pimBroadcastInt(accObj, 0);

    // 2. Execute Spatial Convolution (numRequiredPIMRows is 9 for a 3x3 filter)
    for (int i = 0; i < numRequiredPIMRows; i++)
    {
        int filterVal = filterMatrix[i / filterCols][i % filterCols];

        pimCopyHostToDevice((void *)inputMatrix[i].data(), ifmObj);
        pimBroadcastInt(filterObj, filterVal);

        // Hardware Block 1: OSSM (contains the 4:2 CSA logic)
        pimOSSM(ifmObj, filterObj, destP, numBits);

        // Hardware Block 2: OA Tree Reduction
        // Executed to register latency/energy stats in PIMeval tables.
        pimOATreeReduce(destP, destSum);

        // Hardware Block 3: OFC
        // mathematically we feed 'destP' directly into OFC to preserve the spatial alignment
        // so that the Host CPU's 'aggregateConv' can successfully reduce the channels later.
        pimBitSliceExtract(destP, q_sign, 31);
        pimBitSliceExtract(destP, q_mag, 0);

        pimBroadcastInt(destQ, 0);
        pimBroadcastInt(destQM, 0);

                // Hardware Block 4: Carry-Propagate Accumulation
        pimAdd(accObj, destQ, accObj);
    }

    // 3. Extract correct results to host
    outputMatrix.resize(numRequiredPIMCol);
    pimCopyDeviceToHost(accObj, (void *)outputMatrix.data());

    // Clean up
    pimFree(ifmObj);
    pimFree(filterObj);
    pimFree(destP);
    pimFree(destSum);
    pimFree(destQ);
    pimFree(destQM);
    pimFree(q_sign);
    pimFree(q_mag);
    pimFree(accObj);
}

// =====================================================================
// Convolution Layer Orchestrator matching PIMbench standards
// =====================================================================
// void conv2_ON_CNN(std::vector<std::vector<std::vector<int>>> &inputMatrix,
//                   std::vector<std::vector<std::vector<int>>> &kernelMatrix,
//                   std::vector<std::vector<std::vector<int>>> &resultMatrix,
//                   int stride,
//                   int padding)
// {
//     PimDeviceProperties deviceProp;
//     PimStatus status = pimGetDeviceProperties(&deviceProp);
//     if (status != PIM_OK)
//     {
//         std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
//         exit(1);
//     }

//     uint64_t numCols = deviceProp.numColPerSubarray;
//     uint64_t numRows = deviceProp.numRowPerSubarray;
//     uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) *
//                          uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;

//     int inputDepth = inputMatrix.size();
//     int inputHeight = inputMatrix[0].size();
//     int inputWidth = inputMatrix[0][0].size();
//     int kernelDepth = kernelMatrix.size();
//     int kernelHeight = kernelMatrix[0].size();
//     int kernelWidth = kernelMatrix[0][0].size();

//     int outMatRow = std::floor((inputHeight - kernelHeight) / stride) + 1;
//     int outMatCol = std::floor((inputWidth - kernelWidth) / stride) + 1;
//     int numOfMatPerRow = floor((1.0 * numOfBits) / (outMatRow * outMatCol)) < inputDepth ? floor((1.0 * numOfBits) / (outMatRow * outMatCol)) : inputDepth;
//     if (numOfMatPerRow < 1)
//         numOfMatPerRow = 1;

//     int numOfPIMRow = kernelHeight * kernelWidth;
//     resultMatrix.resize(kernelDepth, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol)));

//     for (int i = 0; i < kernelDepth; i++)
//     {
//         int tempcol = 0;
//         std::vector<int> dstVec(outMatRow * outMatCol);
//         std::vector<int> outVector(outMatRow * outMatCol * inputDepth, 0);

//         for (int j = 0; j < inputDepth; j += numOfMatPerRow)
//         {
//             int matChunk = (numOfMatPerRow + j) <= inputDepth ? (numOfMatPerRow + j) : inputDepth;
//             std::vector<std::vector<int>> mergedMat(numOfPIMRow);

//             for (int k = j; k < matChunk; k++)
//             {
//                 std::vector<std::vector<int>> decompMat;
//                 decomposeMatrix(inputHeight, inputWidth, kernelMatrix[i].size(), kernelMatrix[i][0].size(), stride, 0, inputMatrix[k], decompMat);

//                 for (uint64_t idx = 0; idx < mergedMat.size(); idx++)
//                 {
//                     mergedMat[idx].insert(mergedMat[idx].end(),
//                                           std::make_move_iterator(decompMat[idx].begin()),
//                                           std::make_move_iterator(decompMat[idx].end()));
//                 }
//                 tempcol = mergedMat[0].size();
//             }

//             // Execute ON-CNN Bit-Serial MAC sequence on PIM
//             performConv_ON_CNN(kernelMatrix[i], mergedMat, outVector, numOfPIMRow, tempcol);
//         }

//         int hopSize = outMatCol * outMatRow;
//         aggregateConv(outVector, dstVec, hopSize);

//         int ddx = 0;
//         for (int rdx = 0; rdx < outMatRow; ++rdx)
//         {
//             for (int cdx = 0; cdx < outMatCol; ++cdx)
//             {
//                 resultMatrix[i][rdx][cdx] = dstVec[ddx++];
//             }
//         }
//     }
// }

void conv2_ON_CNN(std::vector<std::vector<std::vector<int>>> &inputMatrix,
                  std::vector<std::vector<std::vector<int>>> &kernelMatrix,
                  std::vector<std::vector<std::vector<int>>> &resultMatrix,
                  int stride,
                  int padding)
{
    PimDeviceProperties deviceProp;
    pimGetDeviceProperties(&deviceProp);

    uint64_t numCols = deviceProp.numColPerSubarray;
    uint64_t numRows = deviceProp.numRowPerSubarray;
    uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) *
                         uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;

    int inputDepth = inputMatrix.size();
    int inputHeight = inputMatrix[0].size();
    int inputWidth = inputMatrix[0][0].size();
    int kernelDepth = kernelMatrix.size();
    int kernelHeight = kernelMatrix[0].size();
    int kernelWidth = kernelMatrix[0][0].size();

    int outMatRow = std::floor((inputHeight - kernelHeight) / stride) + 1;
    int outMatCol = std::floor((inputWidth - kernelWidth) / stride) + 1;
    int numOfMatPerRow = floor((1.0 * numOfBits) / (outMatRow * outMatCol)) < inputDepth ? floor((1.0 * numOfBits) / (outMatRow * outMatCol)) : inputDepth;
    if (numOfMatPerRow < 1)
        numOfMatPerRow = 1;

    int numOfPIMRow = kernelHeight * kernelWidth;

    resultMatrix.resize(kernelDepth, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol, 0)));
    std::vector<int> outVector(outMatRow * outMatCol * inputDepth, 0);

    // --- PROFILING MODE: Only process the FIRST kernel and FIRST chunk ---
    std::vector<std::vector<int>> mergedMat(numOfPIMRow);
    int matChunk = (numOfMatPerRow + 0) <= inputDepth ? (numOfMatPerRow + 0) : inputDepth;

    for (int k = 0; k < matChunk; k++)
    {
        std::vector<std::vector<int>> decompMat;
        decomposeMatrix(inputHeight, inputWidth, kernelMatrix[0].size(), kernelMatrix[0][0].size(), stride, 0, inputMatrix[k], decompMat);

        for (uint64_t idx = 0; idx < mergedMat.size(); idx++)
        {
            mergedMat[idx].insert(mergedMat[idx].end(),
                                  std::make_move_iterator(decompMat[idx].begin()),
                                  std::make_move_iterator(decompMat[idx].end()));
        }
    }

    int tempcol = mergedMat[0].size();

    performConv_ON_CNN(kernelMatrix[0], mergedMat, outVector, numOfPIMRow, tempcol);

    std::cout << " [Profiler] Hardware simulated for 1 chunk. Skipping remaining "
              << (kernelDepth * (inputDepth / numOfMatPerRow)) - 1
              << " chunks to save time." << std::endl;
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