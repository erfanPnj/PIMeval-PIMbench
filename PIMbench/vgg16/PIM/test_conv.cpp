#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <cmath>
#include <algorithm>
#include <getopt.h>
#include "util.h"
#include "utilML.h"
#include <iomanip>
#include <chrono>
#include <cassert>

using namespace std;

typedef struct Params
{
  char *dramConfigFile;
  char *imageInputFile;
  char *kernelMatrixFile;
  char *expectedOutputFile;
  char *metaFile;
  bool moreDebugPrints;
} Params;

void usage()
{
  fprintf(stderr,
          "\nUsage:  ./test_conv.out [options]"
          "\n    -i    input image flat text file (test_input.txt)"
          "\n    -k    input kernel flat text file (test_weight.txt)"
          "\n    -e    expected output flat text file (test_output_expected.txt)"
          "\n    -d    dimensions file written by gen.py (default test_meta.txt)"
          "\n    -c    input file containing dramsim config"
          "\n    -m    enable more debug prints (t = true)"
          "\n");
}

struct Params getInputParams(int argc, char **argv)
{
  struct Params p;
  p.dramConfigFile = nullptr;
  p.imageInputFile = nullptr;
  p.kernelMatrixFile = nullptr;
  p.expectedOutputFile = nullptr;
  p.metaFile = (char *)"test_meta.txt";
  p.moreDebugPrints = false;

  int opt;
  while ((opt = getopt(argc, argv, "c:i:k:e:d:m:")) >= 0)
  {
    switch (opt)
    {
    case 'c': p.dramConfigFile = optarg; break;
    case 'i': p.imageInputFile = optarg; break;
    case 'k': p.kernelMatrixFile = optarg; break;
    case 'e': p.expectedOutputFile = optarg; break;
    case 'd': p.metaFile = optarg; break;
    case 'm': p.moreDebugPrints = (*optarg == 't'); break;
    default:
      fprintf(stderr, "\nUnrecognized option!\n");
      usage();
      exit(0);
    }
  }
  return p;
}

std::vector<int> readFlatFile(const std::string &filename)
{
  std::vector<int> data;
  std::ifstream file(filename);
  if (!file.is_open())
  {
    std::cerr << "Error: Could not open file " << filename << std::endl;
    return data;
  }
  int val;
  while (file >> val)
    data.push_back(val);
  return data;
}

int main(int argc, char *argv[])
{
  struct Params params = getInputParams(argc, argv);

  if (params.imageInputFile == nullptr || params.kernelMatrixFile == nullptr || params.expectedOutputFile == nullptr)
  {
    std::cerr << "Error: Missing input files. Please provide -i, -k, and -e arguments.\n";
    usage();
    return 1;
  }

  // Dimensions come from the generator, so Python and C++ cannot disagree
  int C_in, C_out, H, W, KH, KW, stride, padding;
  {
    std::ifstream m(params.metaFile);
    m >> C_in >> C_out >> H >> W >> KH >> KW >> stride >> padding;
    if (!m)
    {
      std::cerr << "Error: cannot read dimensions from " << params.metaFile << std::endl;
      return 1;
    }
  }
  const int outH = (H + 2 * padding - KH) / stride + 1;
  const int outW = (W + 2 * padding - KW) / stride + 1;

  std::vector<int> flat_input = readFlatFile(params.imageInputFile);
  std::vector<int> flat_weight = readFlatFile(params.kernelMatrixFile);
  std::vector<int> expectedOutputFlat = readFlatFile(params.expectedOutputFile);

  if (flat_input.size() != (size_t)C_in * H * W ||
      flat_weight.size() != (size_t)C_out * C_in * KH * KW ||
      expectedOutputFlat.size() != (size_t)C_out * outH * outW)
  {
    std::cerr << "Error: file sizes do not match test_meta.txt (generator and test disagree)\n";
    return 1;
  }

  std::vector<std::vector<std::vector<int>>> inputMatrix(C_in, std::vector<std::vector<int>>(H, std::vector<int>(W)));
  std::vector<std::vector<std::vector<std::vector<int>>>> kernel4D(
      C_out, std::vector<std::vector<std::vector<int>>>(C_in, std::vector<std::vector<int>>(KH, std::vector<int>(KW))));

  int idx = 0;
  for (int c = 0; c < C_in; ++c)
    for (int h = 0; h < H; ++h)
      for (int w = 0; w < W; ++w)
        inputMatrix[c][h][w] = flat_input[idx++];

  idx = 0;
  for (int co = 0; co < C_out; ++co)
    for (int ci = 0; ci < C_in; ++ci)
      for (int h = 0; h < KH; ++h)
        for (int w = 0; w < KW; ++w)
          kernel4D[co][ci][h][w] = flat_weight[idx++];

  if (!createDevice(params.dramConfigFile))
    return 1;

  std::vector<std::vector<std::vector<int>>> resultMatrix;

  std::cout << "........starting test-conv........\n";
  if (params.moreDebugPrints)
  {
    std::cout << "Input [" << C_in << "][" << H << "][" << W << "]  Kernel [" << C_out << "][" << C_in << "]["
              << KH << "][" << KW << "]  stride " << stride << " pad " << padding << "\n";
  }

  auto start = std::chrono::high_resolution_clock::now();
  conv2(inputMatrix, kernel4D, resultMatrix, stride, padding);
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);
  std::cout << "........ending test-conv........\n";

  // Verification
  std::cout << "........verifying results........\n";
  int errors = 0;
  if ((int)resultMatrix.size() != C_out || (int)resultMatrix[0].size() != outH || (int)resultMatrix[0][0].size() != outW)
  {
    std::cout << "FAILED: result shape [" << resultMatrix.size() << "][" << resultMatrix[0].size() << "]["
              << resultMatrix[0][0].size() << "] != expected [" << C_out << "][" << outH << "][" << outW << "]\n";
    errors = 1;
  }
  else
  {
    idx = 0;
    for (int c = 0; c < C_out; ++c)
      for (int h = 0; h < outH; ++h)
        for (int w = 0; w < outW; ++w)
        {
          int expected = expectedOutputFlat[idx++];
          int actual = resultMatrix[c][h][w];
          if (actual != expected)
          {
            if (errors < 10)
              std::cout << "Mismatch at [c=" << c << ", h=" << h << ", w=" << w << "]: "
                        << "Expected=" << expected << " | Actual=" << actual << std::endl;
            errors++;
          }
        }
  }

  if (errors == 0)
    std::cout << "\nSUCCESS: ON-CNN implementation matches PyTorch output bit-for-bit!\n\n";
  else
    std::cout << "\nFAILED: Found " << errors << " mismatches.\n\n";

  pimShowStats();
  std::cout << "Host elapsed time: " << std::fixed << std::setprecision(3) << hostElapsedTime.count() << " ms." << std::endl;

  return errors == 0 ? 0 : 1;
}