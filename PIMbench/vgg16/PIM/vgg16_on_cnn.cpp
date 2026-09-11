// Test: VGG16 with ON-CNN Hardware Accelerator Architecture
// Copyright (c) 2024
// This file is licensed under the MIT License.
//
// Description:
// This implementation integrates the ON-CNN architecture into the full VGG-16 pipeline.
// It leverages a functional-timing simulation model:
// 1. Timing Model: Simulates the ON-CNN PIM APIs (OSSM, OFC, OATree) to accurately
//    rack up cycles and energy in the PIMeval framework for Convolutional layers.
// 2. Functional Model: Computes the exact MAC operations on the CPU so that the
//    subsequent layers (ReLU, MaxPool, Dense) receive bit-accurate values to
//    successfully classify the input image at the final Softmax layer.

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

// =====================================================================
// Parameters Structure & Parser
// =====================================================================
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
            "\n    -k    input csv file containing the kernel matrices"
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
// ON-CNN Convolutional Layer Execution Logic
// Replaces standard conv2 for ON-CNN timing and energy modeling
// =====================================================================
void conv2_on_cnn(const std::vector<std::vector<std::vector<int>>> &inputMatrix,
                  const std::vector<std::vector<std::vector<int>>> &kernelMatrix,
                  std::vector<std::vector<std::vector<int>>> &resultMatrix,
                  int stride, int padding_not_used, bool shouldVerify)
{
    int C_in = inputMatrix.size();
    int H_in = inputMatrix[0].size();
    int W_in = inputMatrix[0][0].size();

    int C_out = kernelMatrix.size();
    int K = kernelMatrix[0].size(); // Assuming square KxK

    // Note: The inputMatrix is already padded before this function is called
    int H_out = (H_in - K) / stride + 1;
    int W_out = (W_in - K) / stride + 1;

    // -----------------------------------------------------------------
    // 1. TIMING & ENERGY MODEL (ON-CNN Architecture Simulation)
    // -----------------------------------------------------------------
    int num_pes = 4096; // 16 tiles * 256 PEs per tile (as per ON-CNN paper)
    int M = 16;         // Multipliers per PE
    int numBits = 16;   // Standard 16-bit precision

    int total_windows = C_out * H_out * W_out;
    int pe_iterations = std::ceil((double)total_windows / num_pes);

    int macs_per_window = C_in * K * K;
    int msdf_iterations = std::ceil((double)macs_per_window / M);

    // Dummy allocations purely for hardware performance metrics logging
    PimObjId srcX = pimAlloc(PIM_ALLOC_V1, num_pes, PIM_INT16);
    PimObjId srcY = pimAllocAssociated(srcX, PIM_INT16);
    PimObjId destP = pimAllocAssociated(srcX, PIM_INT32);
    PimObjId destSum = pimAllocAssociated(srcX, PIM_INT32);
    PimObjId q_mag = pimAllocAssociated(srcX, PIM_BOOL);
    PimObjId q_sign = pimAllocAssociated(srcX, PIM_BOOL);
    PimObjId destQ = pimAllocAssociated(srcX, PIM_INT32);
    PimObjId destQM = pimAllocAssociated(srcX, PIM_INT32);
    PimObjId acc = pimAllocAssociated(srcX, PIM_INT32);

    pimBroadcastUInt(destQ, 0);
    pimBroadcastUInt(destQM, 0);
    pimBroadcastUInt(acc, 0);

    std::vector<int16_t> dummy_data(num_pes, 0);

    for (int pe_it = 0; pe_it < pe_iterations; ++pe_it)
    {
        for (int msdf_it = 0; msdf_it < msdf_iterations; ++msdf_it)
        {

            // Simulating Host-to-Device data movement required for this MSDF chunk
            pimCopyHostToDevice((void *)dummy_data.data(), srcX);
            pimCopyHostToDevice((void *)dummy_data.data(), srcY);

            // Trigger ON-CNN API functions to register hardware costs
            pimOSSM(srcX, srcY, destP, numBits);
            pimOATreeReduce(destP, destSum);

            // Exract dummy condition bits for OFC conditional execution
            pimBitSliceExtract(destSum, q_sign, 31);
            pimBitSliceExtract(destSum, q_mag, 0);

            // On-the-Fly Converter delay is numBits + 5
            for (int step = 0; step < numBits + 5; ++step)
            {
                pimOFC(q_mag, q_sign, destQ, destQM, step);
            }

            // Final Accumulator (CPA)
            pimAdd(acc, destQ, acc);
        }
    }

    // Clean up PIM structures
    pimFree(srcX);
    pimFree(srcY);
    pimFree(destP);
    pimFree(destSum);
    pimFree(q_mag);
    pimFree(q_sign);
    pimFree(destQ);
    pimFree(destQM);
    pimFree(acc);

    // -----------------------------------------------------------------
    // 2. FUNCTIONAL MODEL (CPU Execution for accuracy & progression)
    // -----------------------------------------------------------------
    resultMatrix.assign(C_out, std::vector<std::vector<int>>(H_out, std::vector<int>(W_out, 0)));

    if (shouldVerify)
    {
#pragma omp parallel for collapse(3)
        for (int c_o = 0; c_o < C_out; ++c_o)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int sum = 0;
                    for (int c_i = 0; c_i < C_in; ++c_i)
                    {
                        for (int kh = 0; kh < K; ++kh)
                        {
                            for (int kw = 0; kw < K; ++kw)
                            {
                                sum += inputMatrix[c_i][h * stride + kh][w * stride + kw] * kernelMatrix[c_o][kh][kw];
                            }
                        }
                    }
                    resultMatrix[c_o][h][w] = sum;
                }
            }
        }
    }
}

// =====================================================================
// Main Execution
// =====================================================================
int main(int argc, char *argv[])
{
    struct Params params = getInputParams(argc, argv);
    std::vector<std::vector<std::vector<float>>> inputMatrix_f;
    std::vector<std::vector<std::vector<float>>> kernelMatrix_f;
    std::vector<std::vector<std::vector<int>>> inputMatrix;
    std::vector<std::vector<std::vector<int>>> kernelMatrix;

    // Dimensions of the input image
    int imageHeight = 224;
    int imageWidth = 224;
    int imageDepth = 3;
    // Dimensions of the kernel in the first convolutional layer
    int KernelHeight = 3;
    int kernelWidth = 3;
    // Padding for the input image
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
            assert(depth == imageDepth && "Given input image depth does not match with the expected image depth");
        }
        inputMatrix.resize(depth);
        for (int d = 0; d < depth; ++d)
        {
            addPadding(imageHeight, imageWidth, padding, inputMatrixBeforePadding[d], inputMatrix[d]);
        }
#endif
    }

    // Device Initialization
    PimStatus status;
    if (params.dramConfigFile != nullptr)
    {
        status = pimCreateDeviceFromConfig(PIM_DEVICE_BITSIMD_V_AP, params.dramConfigFile);
    }
    else
    {
        status = pimCreateDevice(PIM_DEVICE_BITSIMD_V_AP, 1, 128, 32, 1024, 8192, 0);
    }

    if (status != PIM_OK)
    {
        cerr << "Failed to create PIM device." << endl;
        return 1;
    }

    std::vector<std::vector<std::vector<int>>> resultMatrix1;
    std::vector<std::vector<std::vector<int>>> resultMatrix2;

    // -------------------------------------------------------------
    // VGG-16 Layer Execution using ON-CNN Timing Model
    // -------------------------------------------------------------

    // conv1-1
    if (params.kernelMatrixFile == nullptr)
    {
        kernelMatrix.resize(64);
        for (auto &mat : kernelMatrix)
            getMatrix(3, 3, 0, mat);
    }
    else
    {
        kernelMatrix_f = read_conv_layer_weights_from_csv(params.kernelMatrixFile, "features.0.weight");
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    std::cout << "........starting conv1-1........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(resultMatrix1.size());
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(224, 224, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv1-2........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

    std::cout << "........starting pooling........\n";
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(resultMatrix2.size());
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
        addPadding(112, 112, 1, resultMatrix2[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv2-1........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(128);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(112, 112, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv2-2........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

    std::cout << "........starting pooling........\n";
    resultMatrix2.clear();
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(128);
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
        addPadding(56, 56, 1, resultMatrix2[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv3-1........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(256);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv3-2........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(256);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(56, 56, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv3-3........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

    std::cout << "........starting pooling........\n";
    resultMatrix2.clear();
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(256);
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
        addPadding(28, 28, 1, resultMatrix2[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv4-1........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv4-2........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(28, 28, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv4-3........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

    std::cout << "........starting pooling........\n";
    resultMatrix2.clear();
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix2.size(); ++i)
        addPadding(14, 14, 1, resultMatrix2[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv5-1........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv5-2........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

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
        kernelMatrix = (params.shouldVerify) ? floatToFixed(kernelMatrix_f) : binarizeMatrix(kernelMatrix_f);
    }
    inputMatrix.clear();
    inputMatrix.resize(512);
    for (uint64_t i = 0; i < resultMatrix1.size(); ++i)
        addPadding(14, 14, 1, resultMatrix1[i], inputMatrix[i]);
    resultMatrix1.clear();
    std::cout << "........starting conv5-3........\n";
    conv2_on_cnn(inputMatrix, kernelMatrix, resultMatrix1, 1, 1, params.shouldVerify);
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);

    std::cout << "........starting pooling........\n";
    resultMatrix2.clear();
    pool(resultMatrix1, 2, 2, 2, resultMatrix2);

    // =====================================================================
    // DENSE LAYERS (Run as usual using standard PIM API or Host logic)
    // =====================================================================

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
        denseWeight = (params.shouldVerify) ? floatToFixed(denseWeight_f) : binarizeMatrix(denseWeight_f);
    }
    std::cout << "........starting dense1........\n";
    gemv(4096, 25088, flattenedMat, denseWeight, denseOutput1);
    std::cout << "........starting RELU........\n";
    performRelu(denseOutput1);

    denseWeight.clear();
    std::vector<int> denseOutput2;
    if (params.kernelMatrixFile == nullptr)
    {
        getMatrix(4096, 4096, 0, denseWeight);
    }
    else
    {
        denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.3.weight");
        denseWeight = (params.shouldVerify) ? floatToFixed(denseWeight_f) : binarizeMatrix(denseWeight_f);
    }
    std::cout << "........starting dense2........\n";
    gemv(4096, 4096, denseOutput1, denseWeight, denseOutput2);
    std::cout << "........starting RELU........\n";
    performRelu(denseOutput2);

    denseWeight.clear();
    std::vector<int> denseOutput3;
    if (params.kernelMatrixFile == nullptr)
    {
        getMatrix(4096, 1000, 0, denseWeight);
    }
    else
    {
        denseWeight_f = read_dense_layer_weights_from_csv(params.kernelMatrixFile, "classifier.6.weight");
        denseWeight = (params.shouldVerify) ? floatToFixed(denseWeight_f) : binarizeMatrix(denseWeight_f);
    }
    std::cout << "........starting dense3........\n";
    gemv(1000, 4096, denseOutput2, denseWeight, denseOutput3);

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