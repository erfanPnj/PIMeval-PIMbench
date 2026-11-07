// Test: C++ version of vgg16
// Copyright (c) 2024 University of Virginia
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
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

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
          "\nUsage:  ./vgg16.out [options]"
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

// ---------------------------------------------------------------------------------------------
// Real VGG16 has 4D convolution filters: [Cout][Cin][3][3]. Every output channel owns a separate
// 3x3 kernel for EVERY input channel. The helpers below build / load / convert such 4D filters.
// ---------------------------------------------------------------------------------------------
typedef std::vector<std::vector<std::vector<std::vector<float>>>> Kernel4DFloat;
typedef std::vector<std::vector<std::vector<std::vector<int>>>> Kernel4DInt;

// Random small signed weights in [-2, 2] (used when no weight file is given).
Kernel4DInt makeRandomKernel4D(int outChannels, int inChannels, int kernelSize, unsigned seed)
{
  std::mt19937 generator(seed);
  std::uniform_int_distribution<int> distribution(-2, 2);
  Kernel4DInt kernel(outChannels, std::vector<std::vector<std::vector<int>>>(
                                      inChannels, std::vector<std::vector<int>>(kernelSize, std::vector<int>(kernelSize, 0))));
  for (int o = 0; o < outChannels; ++o)
    for (int i = 0; i < inChannels; ++i)
      for (int r = 0; r < kernelSize; ++r)
        for (int c = 0; c < kernelSize; ++c)
          kernel[o][i][r][c] = distribution(generator);
  return kernel;
}

// Reads one conv layer from the CSV written by vgg16_weights.py.
// CSV line format:  name, Cout, Cin, rows, cols, then Cout*Cin*rows*cols weights in [o][i][r][c] order.
// The file must contain the FULL 4D tensor; the shape is checked against the expected VGG16 shape.
Kernel4DFloat readConvWeights4DFromCsv(const std::string &filename, const std::string &layerName,
                                       int expectedOutChannels, int expectedInChannels)
{
  std::ifstream file(filename);
  std::string line;
  while (std::getline(file, line))
  {
    std::stringstream lineStream(line);
    std::string item;
    std::vector<std::string> items;
    while (std::getline(lineStream, item, ','))
    {
      items.push_back(item);
    }
    if (items.empty() || items[0] != layerName)
    {
      continue;
    }
    const int outChannels = std::stoi(items[1]);
    const int inChannels = std::stoi(items[2]);
    const int rows = std::stoi(items[3]);
    const int cols = std::stoi(items[4]);
    if (outChannels != expectedOutChannels || inChannels != expectedInChannels || rows != 3 || cols != 3)
    {
      throw std::runtime_error("Unexpected filter shape for " + layerName + " in the CSV file (need Cout x Cin x 3 x 3)");
    }
    if (items.size() != 5 + (size_t)outChannels * inChannels * rows * cols)
    {
      throw std::runtime_error("CSV row of " + layerName + " does not hold the full 4D weight tensor");
    }
    Kernel4DFloat kernel(outChannels, std::vector<std::vector<std::vector<float>>>(
                                          inChannels, std::vector<std::vector<float>>(rows, std::vector<float>(cols, 0.0f))));
    size_t index = 5;
    for (int o = 0; o < outChannels; ++o)
      for (int i = 0; i < inChannels; ++i)
        for (int r = 0; r < rows; ++r)
          for (int c = 0; c < cols; ++c)
            kernel[o][i][r][c] = std::stof(items[index++]);
    return kernel;
  }
  throw std::runtime_error("Layer not found in the CSV file: " + layerName);
}

// Float -> fixed point (same scaling as floatToFixed in utilFixedPoint.h) when verifying,
// otherwise 0/1 binarization (same as binarizeMatrix in utilML.h).
Kernel4DInt convertKernel4D(const Kernel4DFloat &kernelFloat, bool useFixedPoint)
{
  Kernel4DInt kernel(kernelFloat.size());
  for (size_t o = 0; o < kernelFloat.size(); ++o)
  {
    kernel[o].resize(kernelFloat[o].size());
    for (size_t i = 0; i < kernelFloat[o].size(); ++i)
    {
      kernel[o][i].resize(kernelFloat[o][i].size());
      for (size_t r = 0; r < kernelFloat[o][i].size(); ++r)
      {
        kernel[o][i][r].resize(kernelFloat[o][i][r].size());
        for (size_t c = 0; c < kernelFloat[o][i][r].size(); ++c)
        {
          const float w = kernelFloat[o][i][r][c];
          kernel[o][i][r][c] = useFixedPoint ? static_cast<int>(w * FIXED_POINT_SCALING_FACTOR) : (w > 0 ? 1 : 0);
        }
      }
    }
  }
  return kernel;
}

// The 13 convolution layers of VGG16 (all 3x3, stride 1, padding 1).
struct ConvLayerSpec
{
  const char *name;    // printed name
  const char *csvName; // layer name in vgg16_weights.csv
  int inChannels;      // Cin
  int outChannels;     // Cout
  int inputSize;       // height = width of the (unpadded) input feature map
  bool poolAfter;      // 2x2 max pool after this layer's ReLU
};

static const ConvLayerSpec kVgg16ConvLayers[13] = {
    {"conv1-1", "features.0.weight", 3, 64, 224, false},
    {"conv1-2", "features.2.weight", 64, 64, 224, true},
    {"conv2-1", "features.5.weight", 64, 128, 112, false},
    {"conv2-2", "features.7.weight", 128, 128, 112, true},
    {"conv3-1", "features.10.weight", 128, 256, 56, false},
    {"conv3-2", "features.12.weight", 256, 256, 56, false},
    {"conv3-3", "features.14.weight", 256, 256, 56, true},
    {"conv4-1", "features.17.weight", 256, 512, 28, false},
    {"conv4-2", "features.19.weight", 512, 512, 28, false},
    {"conv4-3", "features.21.weight", 512, 512, 28, true},
    {"conv5-1", "features.24.weight", 512, 512, 14, false},
    {"conv5-2", "features.26.weight", 512, 512, 14, false},
    {"conv5-3", "features.28.weight", 512, 512, 14, true},
};

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  std::vector<std::vector<std::vector<int>>> inputMatrix; // padded input of the current conv layer
  // Dimensions of the input image
  int imageHeight = 224;
  int imageWidth = 224;
  int imageDepth = 3;
  // Padding for every conv layer
  int padding = 1;

  if (params.imageInputFile == nullptr)
  {
    inputMatrix.resize(imageDepth);
    for (int i = 0; i < imageDepth; i++)
    {
      getMatrix(imageHeight, imageWidth, padding, inputMatrix[i]);
    }
  }
  else // Get inputMatrix from the input image
  {
#ifdef COMPILE_WITH_JPEG
    // If JPEG lib is not supported, below code will not work. In that case disable JPEG by adding COMPILE_WITH_JPEG=0 during make.
    std::string outputFile = "resized_output.jpg";
    std::vector<std::vector<std::vector<int>>> inputMatrixBeforePadding;
    readJPEG(params.imageInputFile, inputMatrixBeforePadding, imageHeight, imageWidth);
    writeResizedImage(outputFile, inputMatrixBeforePadding);
    int depth = inputMatrixBeforePadding.size();
    if (depth != imageDepth)
    {
      std::cerr << "Assertion failed: depth (" << depth << ") != imageDepth (" << imageDepth << ")\n";
      assert(depth == imageDepth && "Given input image depth does not match with the expected image depth");
    }
    inputMatrix.resize(depth);
    for (int d = 0; d < depth; ++d)
    {
      addPadding(imageHeight, imageWidth, padding, inputMatrixBeforePadding[d], inputMatrix[d]);
    }
#endif
  }

  if (!createDevice(params.dramConfigFile))
    return 1;

  // ---- 13 convolution layers (+ ReLU, + 5 max pools), real 4D filters [Cout][Cin][3][3] ----
  std::vector<std::vector<std::vector<int>>> resultMatrix1; // conv output
  std::vector<std::vector<std::vector<int>>> resultMatrix2; // pooled output
  for (int layer = 0; layer < 13; ++layer)
  {
    const ConvLayerSpec &spec = kVgg16ConvLayers[layer];

    // Build the 4D filters of this layer
    Kernel4DInt kernelMatrix;
    if (params.kernelMatrixFile == nullptr)
    {
      kernelMatrix = makeRandomKernel4D(spec.outChannels, spec.inChannels, 3, 1000 + layer);
    }
    else
    {
      Kernel4DFloat kernelMatrix_f = readConvWeights4DFromCsv(params.kernelMatrixFile, spec.csvName, spec.outChannels, spec.inChannels);
      kernelMatrix = convertKernel4D(kernelMatrix_f, params.shouldVerify);
    }

    // Layers after the first one: pad the previous layer's output
    if (layer > 0)
    {
      std::vector<std::vector<std::vector<int>>> &previous = kVgg16ConvLayers[layer - 1].poolAfter ? resultMatrix2 : resultMatrix1;
      inputMatrix.clear();
      inputMatrix.resize(previous.size());
      for (uint64_t i = 0; i < previous.size(); ++i)
      {
        addPadding(spec.inputSize, spec.inputSize, padding, previous[i], inputMatrix[i]);
      }
      previous.clear();
      previous.shrink_to_fit();
    }

    std::cout << "........starting " << spec.name << "........\n";
    if (params.moreDebugPrints == true)
    {
      std::cout << "Input matrix dimensions after padding: ";
      printMatrixDimensions(inputMatrix);
      std::cout << "Kernel dimensions [Cout][Cin][Hf][Wf]: " << kernelMatrix.size() << " x " << kernelMatrix[0].size()
                << " x " << kernelMatrix[0][0].size() << " x " << kernelMatrix[0][0][0].size() << std::endl;
    }
    resultMatrix1.clear();
    conv2(inputMatrix, kernelMatrix, resultMatrix1, 1, 0); // 4D overload: padding was already added above
    std::cout << "........ending " << spec.name << "........\n";

    // RELU
    std::cout << "........starting RELU........\n";
    relu(resultMatrix1);
    std::cout << "........ending RELU........\n";

    // pool
    if (spec.poolAfter)
    {
      std::cout << "........starting pooling........\n";
      resultMatrix2.clear();
      pool(resultMatrix1, 2, 2, 2, resultMatrix2);
      std::cout << "........ending pooling........\n";
      resultMatrix1.clear();
      resultMatrix1.shrink_to_fit();
    }
  }
  // After conv5-3 the last pool output (512 x 7 x 7) is in resultMatrix2

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
    if (params.shouldVerify == true)
    {
      denseWeight = floatToFixed(denseWeight_f);
    }
    else
    {
      denseWeight = binarizeMatrix(denseWeight_f);
    }
  }
  std::cout << "........starting dense1........\n";
  if (params.moreDebugPrints == true)
  {
    // Check the dimensions of the input and weight matrices
    std::cout << "Input matrix dimensions: " << flattenedMat.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);
  }
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
    if (params.shouldVerify == true)
    {
      denseWeight = floatToFixed(denseWeight_f);
    }
    else
    {
      denseWeight = binarizeMatrix(denseWeight_f);
    }
  }
  std::cout << "........starting dense2........\n";
  if (params.moreDebugPrints == true)
  {
    // Check the dimensions of the input and weight matrices
    std::cout << "Input matrix dimensions: " << denseOutput1.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);
  }
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
    if (params.shouldVerify == true)
    {
      denseWeight = floatToFixed(denseWeight_f);
    }
    else
    {
      denseWeight = binarizeMatrix(denseWeight_f);
    }
  }
  std::cout << "........starting dense3........\n";
  if (params.moreDebugPrints == true)
  {
    // Check the dimensions of the input and weight matrices
    std::cout << "Input matrix dimensions: " << denseOutput2.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);
  }
  gemv(1000, 4096, denseOutput2, denseWeight, denseOutput3);
  std::cout << "........ending dense3........\n";

  // perform softmax in host
  // std::vector<double> resultVector;
  // std::vector<float> denseOutput3_f;
  // denseOutput3_f = fixedToFloat(denseOutput3);
  // auto start = std::chrono::high_resolution_clock::now();
  // if (params.shouldVerify == true) {
  //   softmaxOnHost(denseOutput3_f, resultVector);
  // } else {
  //   softmaxOnHost(denseOutput3, resultVector);
  // }
  // auto end = std::chrono::high_resolution_clock::now();
  // hostElapsedTime += (end - start);
  std::vector<int> resultVector;
  std::vector<float> resultVector_f;
  softMaxPIM(denseOutput3, resultVector);
  resultVector_f = fixedToFloat(resultVector);
  std::cout << "Dimensions of the softmax output: " << resultVector.size() << std::endl;

  // *********************************************************************************************************************************
  //  Verification Process:
  //  The following code initializes a vector of pairs, fills it with the softmax output values and their corresponding indices.
  //  Then sorts the pairs by value in descending order, and prints out the top 5 values along with their indices.
  //  These indices can be used to map back to the original 1000 output classes of the VGG16 model.
  //  The top 5 results are printed as the goal is to see if the correct class label is among the indices of the top 5 values.
  //  This is important in many classification tasks where the top prediction might not always be correct,
  //  but the correct label might still be within the top 5 highest probability predictions.
  // *********************************************************************************************************************************

  // Create a vector of pairs to store value-index pairs
  std::vector<std::pair<double, int>> valueIndexPairs;
  // Populate the vector with value-index pairs
  for (uint64_t i = 0; i < resultVector.size(); ++i)
  {
    valueIndexPairs.push_back(std::make_pair(resultVector[i], i));
  }
  // Sort the vector of pairs based on values in descending order
  std::sort(valueIndexPairs.begin(), valueIndexPairs.end(), std::greater<std::pair<double, int>>());
  // Print the top 5 values along with their indices
  std::cout << "Top 5 values and corresponding indices:\n";
  for (int i = 0; i < 5; ++i)
  {
    std::cout << "Value: " << valueIndexPairs[i].first << " Index: " << valueIndexPairs[i].second << std::endl;
  }

  pimShowStats();
  cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << endl;

  return 0;
}