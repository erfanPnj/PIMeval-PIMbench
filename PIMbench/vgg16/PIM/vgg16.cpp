// Test: C++ version of vgg16 (True 4D Conv Filters [Cout][Cin][Hf][Wf])
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

using namespace std;

using Tensor3D = std::vector<std::vector<std::vector<int>>>;
using Kernel4D = std::vector<std::vector<std::vector<std::vector<int>>>>;
using Kernel4DFloat = std::vector<std::vector<std::vector<std::vector<float>>>>;

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
  while ((opt = getopt(argc, argv, "h:c:v:i:k:m:")) >= 0)
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

// Read 4D convolutional weights [Cout][Cin][Hf][Wf] from CSV.
// CSV format: layer_name, Cout, Cin, Hf, Wf, w0, w1, ...
Kernel4DFloat read_conv_layer_weights_4d_from_csv(const std::string &filename,
                                                  const std::string &layer_name,
                                                  int expectedCout,
                                                  int expectedCin)
{
  std::ifstream file(filename);
  std::string line;

  while (std::getline(file, line))
  {
    std::stringstream ss(line);
    std::string item;
    std::vector<std::string> items;

    while (std::getline(ss, item, ','))
    {
      items.push_back(item);
    }

    if (!items.empty() && items[0] == layer_name)
    {
      int cout = std::stoi(items[1]);
      int cin = std::stoi(items[2]);
      int rows = std::stoi(items[3]);
      int cols = std::stoi(items[4]);

      if (cout <= 0)
        cout = expectedCout;
      if (cin <= 0)
        cin = expectedCin;

      Kernel4DFloat kernel4d(
          cout, std::vector<std::vector<std::vector<float>>>(
                    cin, std::vector<std::vector<float>>(
                             rows, std::vector<float>(cols, 0.0f))));

      const size_t availWeights = (items.size() > 5) ? (items.size() - 5) : 0;
      const size_t full4DCount = static_cast<size_t>(cout) * cin * rows * cols;
      size_t idx = 5;

      if (availWeights >= full4DCount)
      {
        for (int co = 0; co < cout; ++co)
        {
          for (int ci = 0; ci < cin; ++ci)
          {
            for (int r = 0; r < rows; ++r)
            {
              for (int c = 0; c < cols; ++c)
              {
                kernel4d[co][ci][r][c] = std::stof(items[idx++]);
              }
            }
          }
        }
      }
      else
      {
        // Fallback if CSV only stored 1 channel per filter (Cout * Hf * Wf values)
        for (int co = 0; co < cout; ++co)
        {
          for (int r = 0; r < rows; ++r)
          {
            for (int c = 0; c < cols; ++c)
            {
              float val = (idx < items.size()) ? std::stof(items[idx++]) : 0.0f;
              for (int ci = 0; ci < cin; ++ci)
              {
                kernel4d[co][ci][r][c] = val;
              }
            }
          }
        }
      }
      return kernel4d;
    }
  }
  throw std::runtime_error("Layer not found in the CSV file: " + layer_name);
}

// Build or load a 4D integer kernel [Cout][Cin][Hf][Wf] for a VGG16 conv layer
void buildConvKernel4D(const Params &params,
                       const char *csvLayerName,
                       int cout,
                       int cin,
                       int kh,
                       int kw,
                       Kernel4D &kernel4D)
{
  kernel4D.clear();
  kernel4D.resize(cout, Tensor3D(cin));

  if (params.kernelMatrixFile == nullptr)
  {
    for (int co = 0; co < cout; ++co)
    {
      for (int ci = 0; ci < cin; ++ci)
      {
        getMatrix(kh, kw, 0, kernel4D[co][ci]);
      }
    }
  }
  else
  {
    Kernel4DFloat kernel4D_f =
        read_conv_layer_weights_4d_from_csv(params.kernelMatrixFile, csvLayerName, cout, cin);
    for (int co = 0; co < cout; ++co)
    {
      if (params.shouldVerify)
      {
        kernel4D[co] = floatToFixed(kernel4D_f[co]);
      }
      else
      {
        kernel4D[co] = binarizeMatrix(kernel4D_f[co]);
      }
    }
  }
}

void printKernel4DDimensions(const Kernel4D &k)
{
  if (k.empty() || k[0].empty() || k[0][0].empty())
  {
    std::cout << "0x0x0x0" << std::endl;
    return;
  }
  std::cout << k.size() << "x" << k[0].size() << "x" << k[0][0].size() << "x" << k[0][0][0].size() << std::endl;
}

// Pad each channel of a 3D feature map [Cin][H][W] into [Cin][H + 2*pad][W + 2*pad]
void padFeatureMap3D(int height, int width, int pad, Tensor3D &inMat, Tensor3D &outMat)
{
  outMat.clear();
  outMat.resize(inMat.size());
  for (size_t c = 0; c < inMat.size(); ++c)
  {
    addPadding(height, width, pad, inMat[c], outMat[c]);
  }
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);
  Tensor3D inputMatrix;
  Kernel4D kernelMatrix4D;

  // Dimensions of the input image
  int imageHeight = 224;
  int imageWidth = 224;
  int imageDepth = 3;
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
    Tensor3D inputMatrixBeforePadding;
    readJPEG(params.imageInputFile, inputMatrixBeforePadding, imageHeight, imageWidth);
    writeResizedImage(outputFile, inputMatrixBeforePadding);
    int depth = inputMatrixBeforePadding.size();
    if (depth != imageDepth)
    {
      std::cerr << "Assertion failed: depth (" << depth << ") != imageDepth (" << imageDepth << ")\n";
      assert(depth == imageDepth && "Given input image depth does not match with the expected image depth");
    }
    padFeatureMap3D(imageHeight, imageWidth, padding, inputMatrixBeforePadding, inputMatrix);
#endif
  }

  if (!createDevice(params.dramConfigFile))
    return 1;

  Tensor3D resultMatrix1;
  Tensor3D resultMatrix2;

  // ==================== Block 1 ====================
  // conv1-1: Cin=3, Cout=64, 224x224, 3x3
  buildConvKernel4D(params, "features.0.weight", 64, 3, 3, 3, kernelMatrix4D);
  std::cout << "........starting conv1-1........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv1-1........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv1-2: Cin=64, Cout=64, 224x224, 3x3
  buildConvKernel4D(params, "features.2.weight", 64, 64, 3, 3, kernelMatrix4D);
  padFeatureMap3D(224, 224, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv1-2........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv1-2........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  std::cout << "........starting pooling........\n";
  resultMatrix2.clear();
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // ==================== Block 2 ====================
  // conv2-1: Cin=64, Cout=128, 112x112, 3x3
  buildConvKernel4D(params, "features.5.weight", 128, 64, 3, 3, kernelMatrix4D);
  padFeatureMap3D(112, 112, 1, resultMatrix2, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-1........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv2-1........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv2-2: Cin=128, Cout=128, 112x112, 3x3
  buildConvKernel4D(params, "features.7.weight", 128, 128, 3, 3, kernelMatrix4D);
  padFeatureMap3D(112, 112, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv2-2........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv2-2........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  std::cout << "........starting pooling........\n";
  resultMatrix2.clear();
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // ==================== Block 3 ====================
  // conv3-1: Cin=128, Cout=256, 56x56, 3x3
  buildConvKernel4D(params, "features.10.weight", 256, 128, 3, 3, kernelMatrix4D);
  padFeatureMap3D(56, 56, 1, resultMatrix2, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-1........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv3-1........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-2: Cin=256, Cout=256, 56x56, 3x3
  buildConvKernel4D(params, "features.12.weight", 256, 256, 3, 3, kernelMatrix4D);
  padFeatureMap3D(56, 56, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-2........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv3-2........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv3-3: Cin=256, Cout=256, 56x56, 3x3
  buildConvKernel4D(params, "features.14.weight", 256, 256, 3, 3, kernelMatrix4D);
  padFeatureMap3D(56, 56, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv3-3........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv3-3........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  std::cout << "........starting pooling........\n";
  resultMatrix2.clear();
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // ==================== Block 4 ====================
  // conv4-1: Cin=256, Cout=512, 28x28, 3x3
  buildConvKernel4D(params, "features.17.weight", 512, 256, 3, 3, kernelMatrix4D);
  padFeatureMap3D(28, 28, 1, resultMatrix2, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-1........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv4-1........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-2: Cin=512, Cout=512, 28x28, 3x3
  buildConvKernel4D(params, "features.19.weight", 512, 512, 3, 3, kernelMatrix4D);
  padFeatureMap3D(28, 28, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-2........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv4-2........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv4-3: Cin=512, Cout=512, 28x28, 3x3
  buildConvKernel4D(params, "features.21.weight", 512, 512, 3, 3, kernelMatrix4D);
  padFeatureMap3D(28, 28, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv4-3........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv4-3........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  std::cout << "........starting pooling........\n";
  resultMatrix2.clear();
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // ==================== Block 5 ====================
  // conv5-1: Cin=512, Cout=512, 14x14, 3x3
  buildConvKernel4D(params, "features.24.weight", 512, 512, 3, 3, kernelMatrix4D);
  padFeatureMap3D(14, 14, 1, resultMatrix2, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-1........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv5-1........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-2: Cin=512, Cout=512, 14x14, 3x3
  buildConvKernel4D(params, "features.26.weight", 512, 512, 3, 3, kernelMatrix4D);
  padFeatureMap3D(14, 14, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-2........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv5-2........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  // conv5-3: Cin=512, Cout=512, 14x14, 3x3
  buildConvKernel4D(params, "features.28.weight", 512, 512, 3, 3, kernelMatrix4D);
  padFeatureMap3D(14, 14, 1, resultMatrix1, inputMatrix);
  resultMatrix1.clear();
  resultMatrix1.shrink_to_fit();
  std::cout << "........starting conv5-3........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input matrix dimensions after padding: ";
    printMatrixDimensions(inputMatrix);
    std::cout << "Kernel 4D dimensions (Cout x Cin x KH x KW): ";
    printKernel4DDimensions(kernelMatrix4D);
  }
  conv2(inputMatrix, kernelMatrix4D, resultMatrix1, 1, 0);
  std::cout << "........ending conv5-3........\n";

  std::cout << "........starting RELU........\n";
  relu(resultMatrix1);
  std::cout << "........ending RELU........\n";

  std::cout << "........starting pooling........\n";
  resultMatrix2.clear();
  pool(resultMatrix1, 2, 2, 2, resultMatrix2);
  std::cout << "........ending pooling........\n";

  // Free 4D conv kernel memory before dense layers
  kernelMatrix4D.clear();
  kernelMatrix4D.shrink_to_fit();

  // ==================== Dense Layers ====================
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
    std::cout << "Input matrix dimensions: " << flattenedMat.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);
  }
  gemv(4096, 25088, flattenedMat, denseWeight, denseOutput1);
  std::cout << "........ending dense1........\n";

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
    std::cout << "Input matrix dimensions: " << denseOutput1.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);
  }
  gemv(4096, 4096, denseOutput1, denseWeight, denseOutput2);
  std::cout << "........ending dense2........\n";

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
    std::cout << "Input matrix dimensions: " << denseOutput2.size() << std::endl;
    std::cout << "Weight matrix dimensions: ";
    printMatrixDimensions(denseWeight);
  }
  gemv(1000, 4096, denseOutput2, denseWeight, denseOutput3);
  std::cout << "........ending dense3........\n";

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