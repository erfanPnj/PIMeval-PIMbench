// PIMeval Simulator - Application Utilities for Machine Learning Code.
// The utilies for Conv, MaxPooling, RELU, GEMV are got from their respective cpp files. So, any change in the individual cpp file should also be updated here manually.
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef PIM_FUNC_SIM_APPS_UTIL_ML_H
#define PIM_FUNC_SIM_APPS_UTIL_ML_H

#include <iostream>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <random>
#include <algorithm>
#include <cmath>
#include "libpimeval.h"
#include <map>
#include <fstream>
#include <sstream>
#if defined(COMPILE_WITH_JPEG)
#include <jpeglib.h>
#include <functional>
#endif
using namespace std;

std::chrono::duration<double, std::milli> hostElapsedTime = std::chrono::duration<double, std::milli>::zero();

// Decompose the input matrix by sliding the kernel dimensions (kernelHeight * kernelWidth) along the input matrix with a stride.
// Assume the input matrix is padded.
void decomposeMatrix(int matrixRow, int matrixColumn, int kernelHeight, int kernelWidth, int stride, int padding, const std::vector<std::vector<int>> &inputMatrix, std::vector<std::vector<int>> &decompMatrix)
{
  // Calculate the number of rows and columns for the decomposed matrix
  int numRows = kernelHeight * kernelWidth;
  int numCols = ((matrixRow - kernelHeight + 2 * padding) / stride + 1) * ((matrixColumn - kernelWidth + 2 * padding) / stride + 1);
  // Initialize the decomposed matrix with the correct size
  decompMatrix.resize(numRows, std::vector<int>(numCols, 0));

  int colIdx = 0;
  for (int i = 0; i < (matrixRow + 2 * padding - kernelHeight + 1); i += stride)
  {
    for (int j = 0; j < (matrixColumn + 2 * padding - kernelWidth + 1); j += stride)
    {
      int rowIDX = 0;
      for (int k = i; k < i + kernelHeight; k++)
      {
        for (int l = j; l < j + kernelWidth; l++)
        {
          decompMatrix[rowIDX++][colIdx] = inputMatrix[k][l];
        }
      }
      ++colIdx;
    }
  }
}

void softMaxPIM(const std::vector<int> src, std::vector<int> &dst)
{
  uint64_t vectorLength = src.size();
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (srcObj == -1)
  {
    std::cout << "Function: " << __func__ << "Abort: pimAlloc failed for obj" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)src.data(), srcObj);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for src vector" << std::endl;
    return;
  }

  int32_t max = std::numeric_limits<int32_t>::lowest();
  status = pimRedMax(srcObj, &max);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimRedMax failed" << std::endl;
    return;
  }

  status = pimSubScalar(srcObj, srcObj, max);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimSubScalar failed" << std::endl;
    return;
  }

  dst.resize(vectorLength);
  status = pimCopyDeviceToHost(srcObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for dst vector" << std::endl;
    return;
  }

  auto start = std::chrono::high_resolution_clock::now();
#pragma omp parallel for
  for (size_t i = 0; i < vectorLength; ++i)
  {
    dst[i] = std::exp(static_cast<double>(dst[i]));
  }
  auto end = std::chrono::high_resolution_clock::now();
  hostElapsedTime += (end - start);

  status = pimCopyHostToDevice((void *)dst.data(), srcObj);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for dst vector" << std::endl;
    return;
  }

  int64_t redsum = 0;
  status = pimRedSum(srcObj, &redsum);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimRedSum failed" << std::endl;
    return;
  }

  status = pimDivScalar(srcObj, srcObj, redsum);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimDicScalar failed" << std::endl;
    return;
  }

  status = pimCopyDeviceToHost(srcObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed for dst vector" << std::endl;
    return;
  }

  pimFree(srcObj);
}

// Function to perform softmax operation on Host.
//  -> Find the max value in the input vector
//  -> Compute the exponentials of each (element - max_value) in the vector
//  -> Sum these exponentials
//  -> Normalize each exponential by dividing it by the sum of all exponentials
// To handle the case where the input can be either int or float in the softmaxOnHost function, C++ templates are used.
// Templates allow the function to accept a vector of any numeric type.
template <typename T>
void softmaxOnHost(const std::vector<T> &input, std::vector<double> &output)
{
  // Find the maximum value in the input vector for numerical stability
  T max_input = *std::max_element(input.begin(), input.end());

  // Compute the exponentials of each element (subtracting the max value for stability)
  std::vector<double> exponentials(input.size());

#pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i)
  {
    exponentials[i] = std::exp(static_cast<double>(input[i] - max_input));
  }

  // Compute the sum of exponentials
  double sum_exponentials = 0.0;

#pragma omp parallel for reduction(+ : sum_exponentials)
  for (size_t i = 0; i < exponentials.size(); ++i)
  {
    sum_exponentials += exponentials[i];
  }

  // Compute the softmax values
  output.resize(input.size());

#pragma omp parallel for
  for (size_t i = 0; i < input.size(); ++i)
  {
    output[i] = exponentials[i] / sum_exponentials;
  }
}

// Perform the convolution operation in PIM between the filter matrix and the input matrix.
// The function allocates the necessary PIM objects, broadcasts filter values to PIM objects, and performs
// element-wise multiplication followed by summation to produce the output.
// The summed results are then copied from the PIM (device) to Host.
void performConv(std::vector<std::vector<int>> &filterMatrix, std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix, int numRequiredPIMRows, int numRequiredPIMCol)
{
  std::vector<int> temp;
  PimObjId filterObject = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  if (filterObject == -1)
  {
    std::cout << "Function: " << __func__ << "Abort: pimAlloc failed for obj1" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)outputMatrix.data(), filterObject);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed between filterObjects and outputMatrix at i=" << std::endl;
    return;
  }

  PimObjId matrixObject = pimAllocAssociated(filterObject, PIM_INT32);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimAllocAssociated failed for matrix obj = " << matrixObject << std::endl;
    return;
  }

  int col = filterMatrix[0].size();
  for (uint64_t j = 0; j < inputMatrix.size(); j += numRequiredPIMRows)
  {
    for (int i = 0; i < numRequiredPIMRows; i++)
    {
      status = pimCopyHostToDevice((void *)inputMatrix[i + j].data(), matrixObject);
      if (status != PIM_OK)
      {
        std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice from inputMatrix to matrixObjects at iteration: " << i << std::endl;
        return;
      }

      status = pimScaledAdd(matrixObject, filterObject, filterObject, filterMatrix[i / col][i % col]);
      if (status != PIM_OK)
      {
        std::cout << "Function: " << __func__ << "Abort: pimScaledAdd Failed." << std::endl;
        return;
      }
    }
  }

  outputMatrix.resize(numRequiredPIMCol);

  status = pimCopyDeviceToHost(filterObject, outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyDeviceToHost failed between filterObjects[0] and outputMatrix" << std::endl;
    return;
  }

  pimFree(filterObject);
  pimFree(matrixObject);
}

// ON-CNN Data Path Implementation
// Simulates the specific data path: PISO -> OSSM -> OA Tree -> OFC -> Accumulator
void performConvONCNN(std::vector<std::vector<int>> &filterMatrix, std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix, int numRequiredPIMRows, int numRequiredPIMCol)
{
  // safe initialization of outputMatrix to avoid undefined behavior in case of early return
  outputMatrix.assign(numRequiredPIMCol, 0);

  PimObjId accObject = pimAlloc(PIM_ALLOC_AUTO, numRequiredPIMCol, PIM_INT32);
  PimObjId ifmObject = pimAllocAssociated(accObject, PIM_INT32);
  PimObjId filterObject = pimAllocAssociated(accObject, PIM_INT32);
  PimObjId ossmOutputObj = pimAllocAssociated(accObject, PIM_INT32);
  PimObjId ofcOutputObj = pimAllocAssociated(accObject, PIM_INT32);

  if (accObject == -1 || ifmObject == -1 || filterObject == -1)
  {
    std::cout << "Function: " << __func__ << " Abort: pimAlloc failed" << std::endl;
    return;
  }

  pimCopyHostToDevice((void *)outputMatrix.data(), accObject);

  int row = filterMatrix.size();
  int col = filterMatrix[0].size();
  int filterSize = row * col; // number of elements in the filter matrix

  for (uint64_t j = 0; j < inputMatrix.size(); j += numRequiredPIMRows)
  {
    for (int i = 0; i < numRequiredPIMRows; i++)
    {
      pimCopyHostToDevice((void *)inputMatrix[i + j].data(), ifmObject);

      int f_idx = i % filterSize;
      std::vector<int> currentFilter(numRequiredPIMCol, filterMatrix[f_idx / col][f_idx % col]);
      pimCopyHostToDevice((void *)currentFilter.data(), filterObject);

      pimOSSM(ifmObject, filterObject, ossmOutputObj);
      pimOFC(ossmOutputObj, ofcOutputObj);
      pimAdd(accObject, ofcOutputObj, accObject);
    }
  }

  pimCopyDeviceToHost(accObject, outputMatrix.data());

  pimFree(accObject);
  pimFree(ifmObject);
  pimFree(filterObject);
  pimFree(ossmOutputObj);
  pimFree(ofcOutputObj);
}

// ===================== ON-CNN dataflow (Shafique & Lee, IEEE Access 2024) =====================
//
// HOW THIS CODE MAPS ON-CNN ONTO FULCRUM
//
//   The 4096 PEs of ON-CNN (16 tiles x 16 rows x 16 columns) are laid out as ONE flat row of PEs:
//
//       gridRows = 256 groups ("PE rows"), each group has gridCols = 16 PEs ("PE columns")
//
//       PE number  =  filterRow * 16 + windowColumn
//
//       PEs   0..15   -> filter 0 of the block, IFM windows 0..15 of the block
//       PEs  16..31   -> filter 1 of the block, IFM windows 0..15 of the block
//       ...
//       PEs 4080..4095-> filter 255 of the block, IFM windows 0..15 of the block
//
//   * All 16 PEs of one group get the SAME filter  (like one row of ON-CNN, Fig. 5 of the paper).
//   * PE number c of every group gets the SAME IFM window (like one column of ON-CNN).
//   * The grid is always completely full. PEs that have no real work are IDLE:
//       - IFM window number >= totalWindows  -> its IFM values are all 0
//       - filter number >= Cout              -> its filter values are all 0
//     Idle PEs still run (they multiply by zero), like the paper's PEs that have nothing to do.
//
// HOW DATA IS BROUGHT INTO THE SUBARRAYS
//
//   Every PE consumes the flattened window (Cin * Hf * Wf values) 16 values at a time.
//   One piece of 16 values is called a CHUNK; a window needs numChunks = ceil(Cin*Hf*Wf / 16) chunks.
//
//   Three PIM objects cover the whole PE grid (allocated once per layer):
//       ifmVector    : 4096 PEs x 16 int16 values   [PE0's 16 values | PE1's 16 values | ... | PE4095's 16 values]
//       filterVector : 4096 PEs x 16 int16 values   same PE order, allocated ASSOCIATED with ifmVector
//       accVector    : 4096 int64 values            one accumulator per PE
//   The host fills ifmVector and filterVector for EVERY iteration of Algorithm 3 and copies them in with
//   pimCopyHostToDevice, so every PE holds its own copy of its IFM window and of its filter in its own
//   subarray row (Fulcrum has no broadcast). Then pimONCNNMac multiplies and accumulates.
//   Choose num_col_per_subarray = 256 * (PEs per core) so that each core row holds exactly its PEs' data.
//
//   Loop structure (exactly Algorithm 3):
//       for each WINDOW BLOCK (16 windows):
//         for each FILTER BLOCK (256 filters):
//           zero the accumulators
//           for each CHUNK (16 values): fill ifmVector + filterVector, copy to PIM, one MAC run
//           read the accumulators back
//
// OTHER MODELING NOTES
//   1. Flat packing (Algorithm 3, Eq. 6, 9, 10, Table 4): the window is flattened as
//      position = (kernelRow * Wf + kernelCol) * Cin + channel, so one chunk may cross a kernel pixel.
//   2. Architecture variant (ONCNN_VARIANT=SS|SP, default SS), C_SOP from Eq. 7:
//        SS: dM = 3, clock 1.2 ns -> C_SOP = 47 (48 cycles per MAC run)
//        SP: dM = 2, clock 1.1 ns -> C_SOP = 46 (47 cycles per MAC run)
//   3. Not modeled / done on the host: requantization to int16 (oncnnRequantize), ReLU and pooling
//      (the paper's PPU), tile/column power gating (idle PEs still consume simulated energy),
//      and the on-chip broadcast network itself .

constexpr int kOnCnnLanes = 16; // M: number of multipliers per PE
constexpr int kOnCnnCols = 16;  // C: number of IFM windows processed in parallel

// Read an integer environment variable, or return defaultValue if not set
inline long oncnnEnvLong(const char *name, long defaultValue)
{
  const char *s = std::getenv(name);
  if (s != nullptr && s[0] != '\0')
  {
    return std::atol(s);
  }
  return defaultValue;
}

// Check if ONCNN_VARIANT is set to "SP" (Serial-Parallel). Default is "SS" (Serial-Serial).
inline bool oncnnIsVariantSP()
{
  const char *s = std::getenv("ONCNN_VARIANT");
  if (s == nullptr)
  {
    return false;
  }
  std::string val(s);
  return (val == "SP" || val == "sp");
}

// Number of ON-CNN PEs per PIM core (must match ONCNN_PES_PER_CORE in the perf model, default 4)
inline int oncnnPEsPerCore()
{
  long val = oncnnEnvLong("ONCNN_PES_PER_CORE", 4);
  if (val < 1)
  {
    val = 1;
  }
  return static_cast<int>(val);
}

// Ceiling division: ceil(a / b) for positive integers
inline uint64_t oncnnCeilDiv(uint64_t a, uint64_t b)
{
  return (a + b - 1) / b;
}

// Compute ceil(log2(lanes)) for the adder tree depth (for M = 16 lanes, this returns 4)
inline uint64_t oncnnTreeLevels(uint64_t lanes = kOnCnnLanes)
{
  uint64_t lvl = 0;
  for (uint64_t v = 1; v < lanes; v <<= 1)
  {
    ++lvl;
  }
  return lvl;
}

// Paper Eq. 7: C_SOP = dM + dA*ceil(log2 M) + 2*Pa + ceil(log2 M)
inline uint64_t oncnnCsop()
{
  long defaultDm = 3; // SS default
  if (oncnnIsVariantSP())
  {
    defaultDm = 2; // SP default
  }

  long dM = oncnnEnvLong("ONCNN_DELTA_M", defaultDm);
  if (dM < 0)
    dM = 0;

  long dA = oncnnEnvLong("ONCNN_DELTA_A", 2);
  if (dA < 0)
    dA = 0;

  long Pa = oncnnEnvLong("ONCNN_PA", 16);
  if (Pa < 1)
    Pa = 1;

  uint64_t treeLvl = oncnnTreeLevels(kOnCnnLanes);
  return static_cast<uint64_t>(dM) + static_cast<uint64_t>(dA) * treeLvl + 2ULL * static_cast<uint64_t>(Pa) + treeLvl;
}

// Return clock period in nanoseconds (1.1 ns for SP, 1.2 ns for SS, or overridden by ONCNN_CLK_NS)
inline double oncnnClkNs()
{
  const char *s = std::getenv("ONCNN_CLK_NS");
  if (s != nullptr)
  {
    double customClk = std::atof(s);
    if (customClk > 0.0)
    {
      return customClk;
    }
  }
  if (oncnnIsVariantSP())
  {
    return 1.1;
  }
  return 1.2;
}

// Number of 16-lane chunks (pimONCNNMac calls) per (filter block, window block) pass (Eq. 6)
inline uint64_t oncnnChunksPerPE(uint64_t Cin, uint64_t Hf, uint64_t Wf)
{
  return oncnnCeilDiv(Cin * Hf * Wf, kOnCnnLanes);
}

// Expected pimONCNNMac call count for a layer (Eq. 6)
inline uint64_t oncnnEq10Calls(uint64_t Cout, uint64_t Hout, uint64_t Wout, uint64_t Cin,
                               uint64_t Hf, uint64_t Wf, uint64_t numPEs)
{
  uint64_t cols = std::min<uint64_t>(kOnCnnCols, numPEs);
  uint64_t rowsF = std::max<uint64_t>(1, numPEs / cols); // T x R
  uint64_t filterBlocks = oncnnCeilDiv(Cout, rowsF);
  uint64_t windowBlocks = oncnnCeilDiv(Hout * Wout, cols);
  return filterBlocks * windowBlocks * oncnnChunksPerPE(Cin, Hf, Wf);
}

// Paper Eq. 10 cycles: Comp = ceil(Cout/(T*R)) * ceil(Hout*Wout/C) * ceil((C_SOP+1) * Cin*Hf*Wf / M)
inline uint64_t oncnnEq10Cycles(uint64_t Cout, uint64_t Hout, uint64_t Wout, uint64_t Cin,
                                uint64_t Hf, uint64_t Wf, uint64_t numPEs)
{
  uint64_t cols = std::min<uint64_t>(kOnCnnCols, numPEs);
  uint64_t rowsF = std::max<uint64_t>(1, numPEs / cols); // T x R
  uint64_t outerPasses = oncnnCeilDiv(Cout, rowsF) * oncnnCeilDiv(Hout * Wout, cols);
  return outerPasses * oncnnCeilDiv((oncnnCsop() + 1) * Cin * Hf * Wf, kOnCnnLanes);
}

// Helper struct so we can read either a 3D kernel [Cout][Hf][Wf] (shared across channels)
// or a 4D kernel [Cout][Cin][Hf][Wf] without using C++ lambdas.
struct OncnnKernelAccessor
{
  const std::vector<std::vector<std::vector<int>>> *kernel3D = nullptr;
  const std::vector<std::vector<std::vector<std::vector<int>>>> *kernel4D = nullptr;
  bool is4D = false;
};

// Simple function to fetch a weight value from either the 4D or 3D kernel matrix
inline int oncnnGetWeight(const OncnnKernelAccessor &kernel, int filt, int ch, int kr, int kc)
{
  if (kernel.is4D)
  {
    return (*kernel.kernel4D)[filt][ch][kr][kc];
  }
  else
  {
    // In 3D kernel mode, the same 2D filter is shared across all input channels
    return (*kernel.kernel3D)[filt][kr][kc];
  }
}

// Check if an int fits in 16 bits (-32768..32767) and cast to int16_t
inline int16_t oncnnCheckAndCastToInt16(int v, const char *what)
{
  if (v < INT16_MIN || v > INT16_MAX)
  {
    std::cout << "ON-CNN: " << what << " value " << v << " does not fit in 16 bits" << std::endl;
    std::exit(1);
  }
  return static_cast<int16_t>(v);
}

// Print "[DATA_COPY]" lines when ONCNN_LOG_DETAILS=1
inline bool oncnnLogDetailsEnabled()
{
  return oncnnEnvLong("ONCNN_LOG_DETAILS", 0) > 0;
}

// Everything we need to know about one convolution layer and the PE grid, computed once per layer
struct OncnnLayerPlan
{
  // Input feature map: inputDepth x inputHeight x inputWidth
  int inputDepth = 0; // Cin
  int inputHeight = 0;
  int inputWidth = 0;

  // Filters: kernelDepth x Cin x kernelHeight x kernelWidth
  int kernelDepth = 0; // Cout = number of filters
  int kernelHeight = 0;
  int kernelWidth = 0;

  int stride = 1;
  int padding = 0;

  // Output feature map: one output pixel per (filter, window)
  int outputHeight = 0;
  int outputWidth = 0;
  int totalWindows = 0; // outputHeight * outputWidth

  // The flattened window has windowLength = Cin * Hf * Wf values, consumed 16 per chunk
  int windowLength = 0;
  int numChunks = 0; // ceil(windowLength / 16)

  // The PE grid: gridRows filters in parallel, gridCols windows in parallel
  int gridRows = 0; // 256 for 4096 PEs
  int gridCols = 0; // 16
  int gridPEs = 0;  // gridRows * gridCols

  int numFilterBlocks = 0; // ceil(Cout / gridRows)
  int numWindowBlocks = 0; // ceil(totalWindows / gridCols)
};

// Fill in an OncnnLayerPlan. totalPEs is the number of PEs we are allowed to use (e.g. 4096).
inline OncnnLayerPlan oncnnMakeLayerPlan(const std::vector<std::vector<std::vector<int>>> &inputMatrix,
                                         int kernelDepth, int kernelHeight, int kernelWidth,
                                         int stride, int padding, uint64_t totalPEs)
{
  OncnnLayerPlan plan;
  plan.inputDepth = (int)inputMatrix.size();
  plan.inputHeight = (int)inputMatrix[0].size();
  plan.inputWidth = (int)inputMatrix[0][0].size();
  plan.kernelDepth = kernelDepth;
  plan.kernelHeight = kernelHeight;
  plan.kernelWidth = kernelWidth;
  plan.stride = stride;
  plan.padding = padding;

  plan.outputHeight = (plan.inputHeight + 2 * padding - kernelHeight) / stride + 1;
  plan.outputWidth = (plan.inputWidth + 2 * padding - kernelWidth) / stride + 1;
  plan.totalWindows = plan.outputHeight * plan.outputWidth;

  plan.windowLength = plan.inputDepth * kernelHeight * kernelWidth;
  plan.numChunks = (int)oncnnChunksPerPE(plan.inputDepth, kernelHeight, kernelWidth);

  plan.gridCols = (int)std::min<uint64_t>(kOnCnnCols, totalPEs);
  plan.gridRows = (int)std::max<uint64_t>(1, totalPEs / plan.gridCols);
  plan.gridPEs = plan.gridRows * plan.gridCols;

  plan.numFilterBlocks = (int)oncnnCeilDiv(kernelDepth, plan.gridRows);
  plan.numWindowBlocks = (int)oncnnCeilDiv(plan.totalWindows, plan.gridCols);
  return plan;
}

// Percentile-calibrated, round-to-nearest, saturating requantization to int16 (host-side).
inline int oncnnRequantize(std::vector<std::vector<std::vector<int>>> &x)
{
  std::vector<int> mags; // |v| of non-zero activations
  for (const auto &ch : x)
    for (const auto &row : ch)
      for (int v : row)
        if (v != 0)
          mags.push_back(v < 0 ? -v : v);
  if (mags.empty())
    return 0;

  const char *pe = std::getenv("ONCNN_CALIB_PCT");
  const double pct = (pe && *pe) ? std::atof(pe) : 0.9999;
  const size_t k = std::min(mags.size() - 1, (size_t)(pct * mags.size()));
  std::nth_element(mags.begin(), mags.begin() + k, mags.end());
  const int ref = mags[k];
  int shift = 0;
  while ((ref >> shift) > INT16_MAX)
    ++shift;

  if (shift > 0)
  {
    const int64_t half = int64_t(1) << (shift - 1);
    size_t clipped = 0, zeroed = 0;
    for (auto &ch : x)
      for (auto &row : ch)
        for (int &v : row)
        {
          const int64_t a = v < 0 ? -(int64_t)v : v;
          int64_t q = (a + half) >> shift; // round to nearest
          if (q > INT16_MAX)
          {
            q = INT16_MAX;
            ++clipped;
          } // saturate
          if (a != 0 && q == 0)
            ++zeroed;
          v = (int)(v < 0 ? -q : q);
        }
    if (std::getenv("ONCNN_VERBOSE"))
      std::cout << "ON-CNN requant: shift=" << shift << " clipped=" << clipped
                << " nonzero->0=" << zeroed << std::endl;
  }
  return shift;
}

// Core ON-CNN convolution (Algorithm 3 / Eq. 6 / Eq. 10).
// Steps: requantize -> make the layer plan -> allocate PIM objects -> filter blocks -> window blocks -> tiles.
void conv2ONCNN(const std::vector<std::vector<std::vector<int>>> &inputMatrixIn,
                int kernelDepth, int kernelHeight, int kernelWidth,
                const OncnnKernelAccessor &kernelAccessor,
                std::vector<std::vector<std::vector<int>>> &resultMatrix, int stride, int padding)
{
  std::vector<std::vector<std::vector<int>>> inputMatrix = inputMatrixIn; // local copy
  const int shift = oncnnRequantize(inputMatrix);
  if (shift > 0 && std::getenv("ONCNN_VERBOSE"))
    std::cout << "ON-CNN: activations right-shifted by " << shift << " to fit 16 bits" << std::endl;

  PimDeviceProperties deviceProp;
  if (pimGetDeviceProperties(&deviceProp) != PIM_OK)
  {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    exit(1);
  }
  // 16-lane groups must not straddle a row region (16-bit elements)
  if (deviceProp.numColPerSubarray % (kOnCnnLanes * 16) != 0)
  {
    std::cout << "Abort: numColPerSubarray must be a multiple of 256 for ON-CNN 16-lane groups" << std::endl;
    exit(1);
  }

  // How many PEs do we use? Allow fixing the total (e.g. ONCNN_TOTAL_PES=4096) on a fixed memory config.
  const uint64_t maxHardwarePEs = (uint64_t)deviceProp.numPIMCores * oncnnPEsPerCore();
  const uint64_t targetPEs = (uint64_t)oncnnEnvLong("ONCNN_TOTAL_PES", maxHardwarePEs);
  const uint64_t numPEs = std::min(maxHardwarePEs, targetPEs);

  // PLACEMENT CHECK: the timing model assumes every core holds exactly oncnnPEsPerCore() PEs (one core row = k PEs).
  // PIMeval fills each region with one FULL row, so the data placement only agrees with that assumption when one
  // row holds exactly k PEs: num_col_per_subarray = 256 * k  (256 bits = 16 lanes x 16 bits per PE).
  // Example: k = 4  ->  num_col_per_subarray = 1024  ->  4096 PEs on 1024 cores, 4 PEs per core.
  const uint64_t pesPerRow = deviceProp.numColPerSubarray / (kOnCnnLanes * 16);
  if (pesPerRow != (uint64_t)oncnnPEsPerCore())
  {
    std::cout << "ON-CNN WARNING: one core row holds " << pesPerRow << " PEs (num_col_per_subarray="
              << deviceProp.numColPerSubarray << ") but the timing model assumes ONCNN_PES_PER_CORE="
              << oncnnPEsPerCore() << ". Results stay correct, but data placement and the timing/energy model disagree."
              << " Set num_col_per_subarray = " << (uint64_t)(kOnCnnLanes * 16) * oncnnPEsPerCore() << " to make them match." << std::endl;
  }

  const OncnnLayerPlan plan = oncnnMakeLayerPlan(inputMatrix, kernelDepth, kernelHeight, kernelWidth,
                                                 stride, padding, numPEs);

  resultMatrix.assign(plan.kernelDepth,
                      std::vector<std::vector<int>>(plan.outputHeight, std::vector<int>(plan.outputWidth, 0)));

  // ---------------------------------------------------------------------------------------------
  // Three PIM objects for the whole layer. Each one has one slot of 16 values for EVERY PE:
  //   ifmVector    : 4096 PEs x 16 int16 values  (the 16 IFM values each PE multiplies in this iteration)
  //   filterVector : 4096 PEs x 16 int16 values  (the 16 filter values each PE multiplies in this iteration)
  //   accVector    : 4096 int64 values           (one accumulator per PE, i.e. one per group of 16 lanes)
  // filterVector and accVector are allocated ASSOCIATED with ifmVector, so element i of all three objects
  // sits in the same core, which is what makes the MAC valid.
  // ---------------------------------------------------------------------------------------------
  const uint64_t vectorElements = (uint64_t)plan.gridPEs * kOnCnnLanes;
  PimObjId ifmVector = pimAlloc(PIM_ALLOC_AUTO, vectorElements, PIM_INT16);
  if (ifmVector == -1)
  {
    std::cout << "ON-CNN Abort: pimAlloc failed for ifmVector" << std::endl;
    return;
  }
  PimObjId filterVector = pimAllocAssociated(ifmVector, PIM_INT16);
  PimObjId accVector = pimAllocAssociatedGrouped(ifmVector, PIM_INT64, kOnCnnLanes);
  if (filterVector == -1 || accVector == -1)
  {
    std::cout << "ON-CNN Abort: could not allocate filterVector / accVector" << std::endl;
    pimFree(ifmVector);
    if (filterVector != -1)
      pimFree(filterVector);
    if (accVector != -1)
      pimFree(accVector);
    return;
  }

  // Host-side copies of the three objects. We fill them and then copy them to the PIM objects.
  std::vector<int16_t> ifmHost(vectorElements, 0);
  std::vector<int16_t> filterHost(vectorElements, 0);
  std::vector<int64_t> accHost(plan.gridPEs, 0);
  const std::vector<int64_t> accZeros(plan.gridPEs, 0);

  size_t overflowCount = 0;
  int64_t worstOverflowValue = 0;

  // ---------------------------------------------------------------------------------------------
  // PREPARATION: flatten every IFM window and every filter into a plain list of windowLength values.
  // Flatten order (the same for windows and filters):  position = pixel * Cin + channel,
  //                                                     pixel    = kernelRow * Wf + kernelCol
  // so the list reads: [pixel0 ch0, pixel0 ch1, ..., pixel1 ch0, pixel1 ch1, ...].
  // Because windows and filters use the same order, value k of a window always meets value k of a filter.
  // ---------------------------------------------------------------------------------------------
  const int C = plan.gridCols;                // number of IFM windows processed in parallel (PE columns) = 16
  const int R = plan.gridRows;                // number of filters processed in parallel     (PE rows)    = 256
  const int M = kOnCnnLanes;                  // number of multipliers per PE                              = 16
  const int totalWindows = plan.totalWindows; // Hout * Wout
  const int Cout = plan.kernelDepth;
  const int Cin = plan.inputDepth;
  const int windowLength = plan.windowLength; // Cin * Hf * Wf

  // flatWindows[windowIndex][position]: all values the kernel sees for that output window (0 where it sees padding)
  std::vector<std::vector<int16_t>> flatWindows(totalWindows, std::vector<int16_t>(windowLength, 0));
  for (int windowIndex = 0; windowIndex < totalWindows; ++windowIndex)
  {
    const int outputRow = windowIndex / plan.outputWidth;
    const int outputCol = windowIndex % plan.outputWidth;
    int position = 0;
    for (int kernelRow = 0; kernelRow < plan.kernelHeight; ++kernelRow)
    {
      for (int kernelCol = 0; kernelCol < plan.kernelWidth; ++kernelCol)
      {
        const int inputRow = outputRow * plan.stride + kernelRow - plan.padding;
        const int inputCol = outputCol * plan.stride + kernelCol - plan.padding;
        const bool insideImage = (inputRow >= 0 && inputRow < plan.inputHeight && inputCol >= 0 && inputCol < plan.inputWidth);
        for (int channel = 0; channel < Cin; ++channel)
        {
          if (insideImage)
          {
            flatWindows[windowIndex][position] = oncnnCheckAndCastToInt16(inputMatrix[channel][inputRow][inputCol], "activation");
          }
          position++; // padding stays 0
        }
      }
    }
  }

  // flatFilters[filterIndex][position]: all weights of that filter, in the same order
  std::vector<std::vector<int16_t>> flatFilters(Cout, std::vector<int16_t>(windowLength, 0));
  for (int filterIndex = 0; filterIndex < Cout; ++filterIndex)
  {
    int position = 0;
    for (int kernelRow = 0; kernelRow < plan.kernelHeight; ++kernelRow)
    {
      for (int kernelCol = 0; kernelCol < plan.kernelWidth; ++kernelCol)
      {
        for (int channel = 0; channel < Cin; ++channel)
        {
          flatFilters[filterIndex][position] =
              oncnnCheckAndCastToInt16(oncnnGetWeight(kernelAccessor, filterIndex, channel, kernelRow, kernelCol), "weight");
          position++;
        }
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Algorithm 3 of the paper, with the paper's names:
  //   for (w = 0; w < Hout*Wout; w += C)
  //     for (f = 0; f < Cout; f += R)
  //       for (c = 0; c < Cin*Hf*Wf; c += M)
  //         OUT[f][w] += IN[c] * W[f][c]
  // ---------------------------------------------------------------------------------------------

  // w = number of the FIRST IFM window of the current window block (the block holds windows w .. w+C-1)
  for (int w = 0; w < totalWindows; w += C)
  {
    // f = number of the FIRST filter of the current filter block (the block holds filters f .. f+R-1)
    for (int f = 0; f < Cout; f += R)
    {
      // Start of one (window block, filter block) pair: all 4096 PEs begin with an accumulator of 0.
      // Each PE will sum up, chunk after chunk, ONE output value OUT[filter][window].
      if (pimCopyHostToDevice((void *)accZeros.data(), accVector, 0, plan.gridPEs) != PIM_OK)
      {
        std::cout << "ON-CNN Abort: resetting the accumulators failed" << std::endl;
        pimFree(ifmVector);
        pimFree(filterVector);
        pimFree(accVector);
        return;
      }

      // c = position in the flattened window of the FIRST value of the current chunk.
      // One iteration = one chunk of M = 16 values = one MAC run of every PE.
      for (int c = 0; c < windowLength; c += M)
      {
        // STEP 1: fill ifmVector and filterVector, PE after PE, 16 values per PE.
        //   PE number = filterRow * C + windowColumn      (filterRow 0..R-1, windowColumn 0..C-1)
        //   PE (filterRow, windowColumn) gets
        //     in ifmVector   : values c .. c+15 of the flattened IFM window number  w + windowColumn  (same window in every row)
        //     in filterVector: values c .. c+15 of the flattened filter number      f + filterRow     (same filter in every column)
        //   A PE with no real window (w + windowColumn >= Hout*Wout) or no real filter (f + filterRow >= Cout)
        //   gets zeros, and so do values behind the end of the window (last chunk).
        for (int filterRow = 0; filterRow < R; ++filterRow)
        {
          const int filterIndex = f + filterRow;
          for (int windowColumn = 0; windowColumn < C; ++windowColumn)
          {
            const int windowIndex = w + windowColumn;
            const int pe = filterRow * C + windowColumn;

            for (int lane = 0; lane < M; ++lane)
            {
              const int position = c + lane; // position inside the flattened window / filter
              int16_t ifmValue = 0;
              int16_t filterValue = 0;
              if (position < windowLength)
              {
                if (windowIndex < totalWindows)
                {
                  ifmValue = flatWindows[windowIndex][position];
                }
                if (filterIndex < Cout)
                {
                  filterValue = flatFilters[filterIndex][position];
                }
              }
              ifmHost[(size_t)pe * M + lane] = ifmValue;
              filterHost[(size_t)pe * M + lane] = filterValue;
            }
          }
        }
        // --- VERIFICATION CHECK INSIDE THE CHUNK LOOP (right before pimCopyHostToDevice) ---
        const int activeFilterRows = std::max(0, std::min(R, Cout - f));
        const size_t firstIdleIdx = (size_t)activeFilterRows * C * M; // 256 when activeFilterRows == 1

        // 1. Check that all idle PE rows (from firstIdleIdx to 4096*16 - 1) are zero-filled
        size_t nonZeroInIdleRows = 0;
        for (size_t idx = firstIdleIdx; idx < vectorElements; ++idx)
        {
          if (filterHost[idx] != 0)
          {
            ++nonZeroInIdleRows;
          }
        }
        if (nonZeroInIdleRows > 0)
        {
          std::cout << "ERROR: Found " << nonZeroInIdleRows << " non-zero elements in idle PE rows!" << std::endl;
        }
        // Print the result for this chunk
        // std::cout << "[DEBUG] Chunk " << c << " | Active Rows: " << activeFilterRows
        //           << " | First Idle Index: " << firstIdleIdx
        //           << " | Non-Zero in Idle Rows: " << nonZeroInIdleRows << std::endl;

        // STEP 2: copy both vectors into the subarrays, then ONE MAC run on all PEs at once:
        //   acc[pe] += IN[c .. c+M-1] . W[f][c .. c+M-1]     (16 multiplications, summed, added to the accumulator)
        // This is the line  OUT[f][w] += IN[c] x W[f][c]  of Algorithm 3, executed for 256 filters x 16 windows x 16 values.
        if (pimCopyHostToDevice((void *)ifmHost.data(), ifmVector, 0, vectorElements) != PIM_OK ||
            pimCopyHostToDevice((void *)filterHost.data(), filterVector, 0, vectorElements) != PIM_OK ||
            pimONCNNMac(ifmVector, filterVector, accVector) != PIM_OK)
        {
          std::cout << "ON-CNN Abort: copy or MAC failed (w=" << w << ", f=" << f << ", c=" << c << ")" << std::endl;
          pimFree(ifmVector);
          pimFree(filterVector);
          pimFree(accVector);
          return;
        }
      } // end of the c loop: every accumulator now holds the complete dot product of its filter and window

      // STEP 3: read the 4096 accumulators back and store them as OUT[f + filterRow][window w + windowColumn].
      // PEs of idle rows (filter number >= Cout) and idle columns (window number >= Hout*Wout) are skipped.
      if (pimCopyDeviceToHost(accVector, (void *)accHost.data(), 0, plan.gridPEs) != PIM_OK)
      {
        std::cout << "ON-CNN Abort: reading the accumulators failed" << std::endl;
        pimFree(ifmVector);
        pimFree(filterVector);
        pimFree(accVector);
        return;
      }
      for (int filterRow = 0; filterRow < R; ++filterRow)
      {
        const int filterIndex = f + filterRow;
        if (filterIndex >= Cout)
        {
          break; // idle PE rows
        }
        for (int windowColumn = 0; windowColumn < C; ++windowColumn)
        {
          const int windowIndex = w + windowColumn;
          if (windowIndex >= totalWindows)
          {
            break; // idle PE columns
          }
          const int64_t value = accHost[filterRow * C + windowColumn];
          if (value < INT32_MIN || value > INT32_MAX)
          {
            ++overflowCount;
            worstOverflowValue = value;
          }
          resultMatrix[filterIndex][windowIndex / plan.outputWidth][windowIndex % plan.outputWidth] = static_cast<int>(value);
        }
      }
    } // end of the f loop
  } // end of the w loop

  pimFree(ifmVector);
  pimFree(filterVector);
  pimFree(accVector);

  // The accumulators are 64 bit, the result matrix holds int. Report values that do not fit.
  if (overflowCount > 0)
  {
    std::cout << "ON-CNN: int32 accumulator overflow count=" << overflowCount
              << " / " << (size_t)plan.kernelDepth * plan.totalWindows << " outputs (sample=" << worstOverflowValue << ")" << std::endl;
    if (oncnnEnvLong("ONCNN_STRICT", 0) != 0)
    {
      std::cout << "ON-CNN Abort: accumulator overflowed int32 with ONCNN_STRICT=1" << std::endl;
      std::exit(1);
    }
  }

  if (std::getenv("ONCNN_PRINT_EQ10"))
  {
    const uint64_t calls = oncnnEq10Calls(plan.kernelDepth, plan.outputHeight, plan.outputWidth, plan.inputDepth,
                                          plan.kernelHeight, plan.kernelWidth, plan.gridPEs);
    const uint64_t cyc = oncnnEq10Cycles(plan.kernelDepth, plan.outputHeight, plan.outputWidth, plan.inputDepth,
                                         plan.kernelHeight, plan.kernelWidth, plan.gridPEs);
    const double clkNs = oncnnClkNs();
    const char *variant = oncnnIsVariantSP() ? "SP" : "SS";
    std::cout << "ON-CNN Eq.10 expected: " << cyc << " cycles = " << cyc * clkNs * 1e-6
              << " ms | calls=" << calls << " (variant=" << variant
              << ", C_SOP=" << oncnnCsop() << ", PEs=" << plan.gridPEs << ", grid " << plan.gridRows << "x" << plan.gridCols << ")"
              << std::endl;
  }
}

// Overload 1: 3D kernelMatrix [Cout][Hf][Wf] (2D kernel per filter shared across channels, used by CSV loaders)
void conv2(std::vector<std::vector<std::vector<int>>> &inputMatrix,
           std::vector<std::vector<std::vector<int>>> &kernelMatrix,
           std::vector<std::vector<std::vector<int>>> &resultMatrix,
           int stride, int padding)
{
  OncnnKernelAccessor accessor;
  accessor.kernel3D = &kernelMatrix;
  accessor.kernel4D = nullptr;
  accessor.is4D = false;

  int kernelDepth = kernelMatrix.size();
  int kernelHeight = kernelMatrix[0].size();
  int kernelWidth = kernelMatrix[0][0].size();

  conv2ONCNN(inputMatrix, kernelDepth, kernelHeight, kernelWidth, accessor, resultMatrix, stride, padding);
}

// Overload 2: True 4D filters [Cout][Cin][Hf][Wf], as in the paper and test suite
void conv2(std::vector<std::vector<std::vector<int>>> &inputMatrix,
           std::vector<std::vector<std::vector<std::vector<int>>>> &kernel4D,
           std::vector<std::vector<std::vector<int>>> &resultMatrix,
           int stride, int padding)
{
  OncnnKernelAccessor accessor;
  accessor.kernel3D = nullptr;
  accessor.kernel4D = &kernel4D;
  accessor.is4D = true;

  int kernelDepth = kernel4D.size();
  int kernelHeight = kernel4D[0][0].size();
  int kernelWidth = kernel4D[0][0][0].size();

  conv2ONCNN(inputMatrix, kernelDepth, kernelHeight, kernelWidth, accessor, resultMatrix, stride, padding);
}

// This should work for bitSIMD or any PIM that requires vertical data layout.
// Perform the Max Pool operation in PIM for the given input matrix.
// The function allocates the necessary PIM objects, copies the data from host to PIM, and performs Max operation.
// The max results from pimObjectList[0] are then copied from the PIM (device) to Host.
void maxPool(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix)
{
  if (inputMatrix.empty())
  {
    return;
  }
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();

  std::vector<PimObjId> pimObjectList(numRows);
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numCols, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Function: " << __func__ << "Abort: pimAlloc failed for obj1" << std::endl;
    return;
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Function: " << __func__ << "Abort: pimAllocAssociated failed for obj at i=" << i << std::endl;
      return;
    }
    pimObjectList[i] = obj;
  }

  for (uint64_t i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << "Abort: pimCopyHostToDevice failed between inputMatrix and pimObjectList at i=" << i << std::endl;
      return;
    }
  }

  for (uint64_t i = 1; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimMax(pimObjectList[0], pimObjectList[i], pimObjectList[0]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << "Abort: pimMax failed between pimObjectList[0] and pimObjectList[i] at i=" << i << std::endl;
      return;
    }
  }
  outputMatrix.resize(numCols);
  PimStatus status = pimCopyDeviceToHost(pimObjectList[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << "Abort: pimCopyDeviceYoHost failed between pimObjectList[0] and outputMatrix" << std::endl;
    return;
  }
  for (auto elem : pimObjectList)
  {
    pimFree(elem);
  }
}

// Function executing the Max Pool between a 3D input matrix based on a kernel dimensions and a given stride.
// For each input depth, the input matrix is decomposed based on the kernel dimensions and stride, merged together.
// The merged matrix is then passed to maxPool() function to peform max pooling on PIM.
// The result (1D vector) from the PIM is then reconstructed back to a 2D final result matrix for each input depth.
void pool(std::vector<std::vector<std::vector<int>>> &inputMatrix, int kernelHeight, int kernelWidth, int stride, std::vector<std::vector<std::vector<int>>> &resultMatrix)
{
  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK)
  {
    std::cerr << "Abort: pimGetDeviceProperties failed" << std::endl;
    exit(1);
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) * uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;

  uint64_t inputDepth = inputMatrix.size();
  uint64_t inputHeight = inputMatrix[0].size();
  uint64_t inputWidth = inputMatrix[0][0].size();
  uint64_t outputHeight = (inputHeight - kernelHeight) / stride + 1;
  uint64_t outputWidth = (inputWidth - kernelWidth) / stride + 1;

  uint64_t numOfPIMRow = kernelHeight * kernelWidth;
  uint64_t numOfPIMColumn = outputHeight * outputWidth;
  uint64_t numOfMatPerRow = floor((1.0 * numOfBits) / numOfPIMColumn) < inputDepth ? floor((1.0 * numOfBits) / (numOfPIMColumn)) : inputDepth;

  resultMatrix.resize(inputDepth, std::vector<std::vector<int>>(outputHeight, std::vector<int>(outputWidth)));

  for (uint64_t i = 0; i < inputDepth; i += 1)
  {
    // This vector packs all the matrices that can be fit into one PIM iteration
    std::vector<std::vector<int>> mergedMat(numOfPIMRow);
    uint64_t matChunk = (numOfMatPerRow + i) <= inputDepth ? (numOfMatPerRow + i) : inputDepth;
    for (uint64_t j = i; j < matChunk; j++)
    {
      std::vector<std::vector<int>> decompMat;
      decomposeMatrix(inputHeight, inputWidth, kernelHeight, kernelWidth, stride, 0, inputMatrix[j], decompMat);
      // Merge the matrices
      for (uint64_t idx = 0; idx < mergedMat.size(); idx++)
      {
        mergedMat[idx].insert(mergedMat[idx].end(),
                              std::make_move_iterator(decompMat[idx].begin()),
                              std::make_move_iterator(decompMat[idx].end()));
      }
    }

    std::vector<int> outVector;
    maxPool(mergedMat, outVector);

    uint64_t idx = 0;
    for (uint64_t j = i; j < matChunk; ++j)
    {
      for (uint64_t r = 0; r < outputHeight; ++r)
      {
        for (uint64_t c = 0; c < outputWidth; ++c)
        {
          resultMatrix[j][r][c] = outVector[idx++];
        }
      }
    }
  }
}

// Performs General Matrix-Vector Multiplication (GEMV) in PIM
// The function computes the matrix-vector product of the source matrix and the source vector, and stores the result in the destination vector.
// The function allocates PIM objects, performs element-wise multiplication, and accumulates the results.
// The accumulated results are then copied from the PIM (device) to the host.
void gemv(uint64_t row, uint64_t col, std::vector<int> &srcVector, std::vector<std::vector<int>> &srcMatrix, std::vector<int> &dst)
{
  PimObjId srcObj = pimAlloc(PIM_ALLOC_AUTO, row, PIM_INT32);
  if (srcObj == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAlloc failed for srcObj" << std::endl;
    return;
  }

  PimObjId dstObj = pimAllocAssociated(srcObj, PIM_INT32);
  if (dstObj == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated failed for dstObj" << std::endl;
    return;
  }

  PimStatus status = pimBroadcastInt(dstObj, 0);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimBroadcastInt failed for dstObj" << std::endl;
    return;
  }

  for (uint64_t i = 0; i < col; ++i)
  {
    status = pimCopyHostToDevice((void *)srcMatrix[i].data(), srcObj);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice failed between srcMatrix and srcObj at i=" << i << std::endl;
      return;
    }

    status = pimScaledAdd(srcObj, dstObj, dstObj, srcVector[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimScaledAdd failed between srcObj1 and srcObj2 at i=" << i << std::endl;
      return;
    }
  }

  dst.resize(row);
  status = pimCopyDeviceToHost(dstObj, (void *)dst.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyDeviceToHost failed between dstObj and dst" << std::endl;
  }
  pimFree(srcObj);
  pimFree(dstObj);
}

// Perform the RELU (REctified Linear Unit) operation, max(0, x), a non-linear activation function in PIM for the given 1D input vector; called after GEMV operation.
// The function allocates the necessary PIM objects, copies the data from host to PIM, and performs Max operation.
// The max results from pimObject are then copied from the PIM (device) to Host.
void performRelu(std::vector<int> &inputVector)
{
  if (inputVector.empty())
  {
    std::cout << "Function: " << __func__ << ", Abort: Input matrix is empty" << std::endl;
    return;
  }
  int numCols = inputVector.size();

  // Initialize reluConst vector with zero for max(0, x) operation.
  std::vector<int> reluConst(numCols, 0);

  PimObjId pimObject = pimAlloc(PIM_ALLOC_AUTO, numCols, PIM_INT32);
  if (pimObject == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAlloc for PimObj pimObject failed" << std::endl;
    return;
  }

  PimObjId RELUConstObj = pimAllocAssociated(pimObject, PIM_INT32);
  if (RELUConstObj == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated for PimObj RELUConstObj failed" << std::endl;
    return;
  }

  PimStatus status = pimCopyHostToDevice((void *)inputVector.data(), pimObject);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from inputMatrix to pimObject failed" << std::endl;
    return;
  }

  status = pimCopyHostToDevice((void *)reluConst.data(), RELUConstObj);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from reluConst to RELUConstObj failed" << std::endl;
    return;
  }

  status = pimMax(RELUConstObj, pimObject, pimObject);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimMax failed between RELUConstObj and pimObject" << std::endl;
    return;
  }

  inputVector.resize(numCols);
  status = pimCopyDeviceToHost(pimObject, inputVector.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyDeviceToHost from pimObject to outputMatrix" << std::endl;
    return;
  }

  pimFree(pimObject);
  pimFree(RELUConstObj);
}

// Perform the RELU (REctified Linear Unit) operation, max(0, x), a non-linear activation function in PIM for the given 2D input matrix.
// The function allocates the necessary PIM objects, copies the data from host to PIM, and performs Max operation.
// The max results from pimObjectList[0] are then copied from the PIM (device) to Host.
void performRelu(const std::vector<std::vector<int>> &inputMatrix, std::vector<int> &outputMatrix)
{
  if (inputMatrix.empty())
  {
    std::cout << "Function: " << __func__ << ", Abort: Input matrix is empty" << std::endl;
    return;
  }
  int numRows = inputMatrix.size();
  int numCols = inputMatrix[0].size();
  // Initialize reluConst vector with zero for max(0, x) operation. Initialize with a different value 'y' for max(y, x) operation.
  std::vector<int> reluConst(numCols, 0);

  std::vector<PimObjId> pimObjectList(numRows);
  PimObjId obj1 = pimAlloc(PIM_ALLOC_AUTO, numCols, PIM_INT32);
  if (obj1 == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAlloc for PimObj obj1 failed" << std::endl;
    return;
  }
  pimObjectList[0] = obj1;
  for (int i = 1; i < numRows; i++)
  {
    PimObjId obj = pimAllocAssociated(pimObjectList[0], PIM_INT32);
    if (obj == -1)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated for PimObj obj failed" << std::endl;
      return;
    }
    pimObjectList[i] = obj;
  }
  PimObjId RELUConstObj = pimAllocAssociated(pimObjectList[0], PIM_INT32);
  if (RELUConstObj == -1)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimAllocAssociated for PimObj RELUConstObj failed" << std::endl;
    return;
  }

  for (uint64_t i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimCopyHostToDevice((void *)inputMatrix[i].data(), pimObjectList[i]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from inputMatrix[" << i << "] to pimObjectList[" << i << "] failed" << std::endl;
      return;
    }
  }
  PimStatus status = pimCopyHostToDevice((void *)reluConst.data(), RELUConstObj);
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyHostToDevice from reluConst to RELUConstObj failed" << std::endl;
    return;
  }

  for (uint64_t i = 0; i < pimObjectList.size(); i++)
  {
    PimStatus status = pimMax(RELUConstObj, pimObjectList[i], pimObjectList[0]);
    if (status != PIM_OK)
    {
      std::cout << "Function: " << __func__ << ", Abort: pimMax failed between RELUConstObj and pimObjectList[" << i << "]" << std::endl;
      return;
    }
  }
  outputMatrix.resize(numCols);
  status = pimCopyDeviceToHost(pimObjectList[0], outputMatrix.data());
  if (status != PIM_OK)
  {
    std::cout << "Function: " << __func__ << ", Abort: pimCopyDeviceToHost from pimObjectList[0] to outputMatrix" << std::endl;
    return;
  }
  for (auto elem : pimObjectList)
  {
    pimFree(elem);
  }
  pimFree(RELUConstObj);
}

// Function executing RELU for a given 3D input matrix.
// For each input depth, the input matrix is decomposed based on the kernel dimensions (1x1) and stride (1), merged together.
// The merged matrix is then passed to performRelu() function to peform RELU on PIM.
// The result (1D vector) from the PIM is then reconstructed back to a 2D final result matrix for each input depth.
void relu(std::vector<std::vector<std::vector<int>>> &inputMatrix)
{

  // Define parameters for processing
  PimDeviceProperties deviceProp;
  PimStatus status = pimGetDeviceProperties(&deviceProp);
  if (status != PIM_OK)
  {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    exit(1);
  }
  // Get the device parameters
  uint64_t numCols = deviceProp.numColPerSubarray;
  uint64_t numRows = deviceProp.numRowPerSubarray;
  uint64_t numOfBits = uint64_t(deviceProp.numRanks) * uint64_t(deviceProp.numBankPerRank) * uint64_t(deviceProp.numSubarrayPerBank) * numCols * numRows;

  // Calculate matrix dimensions
  uint64_t inputDepth = inputMatrix.size();
  uint64_t inputHeight = inputMatrix[0].size();
  uint64_t inputWidth = inputMatrix[0][0].size();

  uint64_t numOfPIMRow = 1;
  unsigned bitsPerElement = 32;
  uint64_t numOfMatPerRow = std::min(static_cast<uint64_t>(std::floor((1.0 * numOfBits) / (inputHeight * inputWidth * bitsPerElement))), static_cast<uint64_t>(inputWidth));

  std::vector<std::vector<int>> decompMat;
  std::vector<std::vector<int>> mergedMat(numOfPIMRow);
  std::vector<int> outVector;
  outVector.resize(inputDepth * inputHeight * inputWidth);

  // Loop through input depth in chunks
  for (uint64_t j = 0; j < inputDepth; j += numOfMatPerRow)
  {
    uint64_t matChunk = (numOfMatPerRow + j) <= inputDepth ? (numOfMatPerRow + j) : inputDepth;
    // Decompose and merge matrices
    for (uint64_t k = j; k < matChunk; k++)
    {
      // 1, 1, 1, 0 in the function call below indicates that the kernel dimensions are 1x1, stride is 1 and padding is 0 while decomposing the matrix for ReLU operation.
      // The kernel dimensions, stride and padding are specified in the functional call to make the function reusable for other operations like max pooling.
      decomposeMatrix(inputHeight, inputWidth, 1, 1, 1, 0, inputMatrix[k], decompMat);
      for (uint64_t idx = 0; idx < mergedMat.size(); idx++)
      {
        mergedMat[idx].reserve(mergedMat[idx].size() + decompMat[idx].size());
        mergedMat[idx].insert(mergedMat[idx].end(), make_move_iterator(decompMat[idx].begin()), make_move_iterator(decompMat[idx].end()));
      }
    }
  }

  performRelu(mergedMat, outVector);

  uint64_t idx = 0;
  for (uint64_t i = 0; i < inputDepth; i += 1)
  {
    for (uint64_t r = 0; r < inputHeight; ++r)
    {
      for (uint64_t c = 0; c < inputWidth; ++c)
      {
        inputMatrix[i][r][c] = outVector[idx++];
      }
    }
  }
}

// Function to read weights of a specific layer from CSV (for convolutional layers).
vector<vector<vector<float>>> read_conv_layer_weights_from_csv(const string &filename, const string &layer_name)
{
  ifstream file(filename); // Open the CSV file
  string line;
  vector<vector<vector<float>>> kernelMatrix; // Initialize a 3D vector to hold the kernel matrix

  while (getline(file, line))
  { // Read the file line by line
    stringstream ss(line);
    string item;
    vector<string> items;

    // Split the line by commas
    while (getline(ss, item, ','))
    {
      items.push_back(item);
    }

    // Check if the layer name matches the current line
    if (items[0] == layer_name)
    {
      int num_kernels = stoi(items[1]); // Depth of the kernel matrix (number of kernels)
      int rows = stoi(items[3]);        // Number of rows in each kernel
      int cols = stoi(items[4]);        // Number of columns in each kernel
      kernelMatrix.resize(num_kernels); // Resize the 3D vector to accommodate the kernels

      int idx = 5; // Start index of the actual weights in the CSV line
      for (int d = 0; d < num_kernels; ++d)
      {                                                       // Loop through each kernel
        kernelMatrix[d].resize(rows, vector<float>(cols, 0)); // Resize each kernel matrix
        for (int r = 0; r < rows; ++r)
        { // Loop through each row
          for (int c = 0; c < cols; ++c)
          {                                                  // Loop through each column
            kernelMatrix[d][r][c] = std::stof(items[idx++]); // Assign the weight value
          }
        }
      }

      return kernelMatrix; // Return the kernel matrix for the specified layer
    }
  }

  throw runtime_error("Layer not found in the CSV file"); // Throw an error if the layer is not found
}

// Function to read weights of a specific layer from CSV (for dense layers).
std::vector<std::vector<float>> read_dense_layer_weights_from_csv(const std::string &filename, const std::string &layer_name)
{
  std::ifstream file(filename);                // Open the CSV file
  std::string line;                            // Variable to hold each line of the CSV
  std::vector<std::vector<float>> denseMatrix; // Matrix to store the weights of the dense layer

  while (std::getline(file, line))
  {
    std::stringstream ss(line);     // Create a string stream from the line
    std::string item;               // Variable to hold each item in the line
    std::vector<std::string> items; // Vector to store all items in the line

    // Split the line by commas
    while (std::getline(ss, item, ','))
    {
      items.push_back(item);
    }

    // Check if the layer name matches the current line
    if (items[0] == layer_name)
    {
      int rows = std::stoi(items[2]);                        // Number of rows in the dense matrix
      int cols = std::stoi(items[1]);                        // Number of columns in the dense matrix
      denseMatrix.resize(rows, std::vector<float>(cols, 0)); // Resize the matrix to the appropriate dimensions

      int idx = 3; // Start index of the actual weights in the CSV line
      // Populate the dense matrix with weights
      for (int r = 0; r < rows; ++r)
      {
        for (int c = 0; c < cols; ++c)
        {
          denseMatrix[r][c] = std::stof(items[idx++]); // Convert the string weight to float and store it in the matrix
        }
      }

      return denseMatrix;
    }
  }

  throw std::runtime_error("Layer not found in the CSV file"); // Throw an error if the layer name is not found
}

// Function to binarize a 3D matrix of floats.
// Added to binarize the values of kernel matrices of the convolutional layers. May not be needed after adding float support.
std::vector<std::vector<std::vector<int>>> binarizeMatrix(const std::vector<std::vector<std::vector<float>>> &weights)
{
  std::vector<std::vector<std::vector<int>>> binarizedMatrix(weights.size());
  for (size_t d = 0; d < weights.size(); ++d)
  {
    binarizedMatrix[d].resize(weights[d].size());
    for (size_t r = 0; r < weights[d].size(); ++r)
    {
      binarizedMatrix[d][r].resize(weights[d][r].size());
      for (size_t c = 0; c < weights[d][r].size(); ++c)
      {
        binarizedMatrix[d][r][c] = (weights[d][r][c] > 0) ? 1 : 0;
      }
    }
  }
  return binarizedMatrix;
}

// Function to binarize a 2D matrix of floats.
// Added to binarize the values of kernel matrices of the dense layers. May not be needed after adding float support.
std::vector<std::vector<int>> binarizeMatrix(const std::vector<std::vector<float>> &weights)
{
  std::vector<std::vector<int>> binarizedMatrix(weights.size());
  for (size_t r = 0; r < weights.size(); ++r)
  {
    binarizedMatrix[r].resize(weights[r].size());
    for (size_t c = 0; c < weights[r].size(); ++c)
    {
      binarizedMatrix[r][c] = (weights[r][c] > 0) ? 1 : 0;
    }
  }
  return binarizedMatrix;
}

#ifdef COMPILE_WITH_JPEG
// Function to read a JPEG image and store its pixel data into inputMatrix.
void readJPEG(const std::string &filename, std::vector<std::vector<std::vector<int>>> &inputMatrix, int &targetHeight, int &targetWidth)
{
  // Open the JPEG file
  FILE *file = fopen(filename.c_str(), "rb");
  if (!file)
  {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  // Initialize the JPEG decompression object
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);

  // Specify the input file
  jpeg_stdio_src(&cinfo, file);

  // Read JPEG header
  jpeg_read_header(&cinfo, TRUE);

  // Set parameters for decompression
  cinfo.out_color_space = JCS_RGB; // RGB color space

  // Start decompression
  jpeg_start_decompress(&cinfo);

  // Allocate memory for storing scanline of decompressed image
  int row_stride = cinfo.output_width * cinfo.output_components;
  JSAMPARRAY buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

  // Calculate scaling factors
  double scaleX = (double)cinfo.output_width / targetWidth;
  double scaleY = (double)cinfo.output_height / targetHeight;

  // Resize the image
  inputMatrix.resize(3, std::vector<std::vector<int>>(targetHeight, std::vector<int>(targetWidth)));

  int row = 0;
  while (cinfo.output_scanline < cinfo.output_height)
  {
    // Read scanline
    jpeg_read_scanlines(&cinfo, buffer, 1);
    if (row < targetHeight)
    {
      // Process each pixel in the scanline
      for (int col = 0; col < targetWidth; ++col)
      {
        int origX = (int)(col * scaleX);
        if (origX >= cinfo.output_width)
          origX = cinfo.output_width - 1;
        for (int color = 0; color < 3; ++color)
        {
          inputMatrix[color][row][col] = buffer[0][origX * cinfo.output_components + color];
        }
      }
    }
    ++row;
  }

  // Finish decompression
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);

  // Close the file
  fclose(file);
}

// Function to write the resized image data to a JPEG file.
// Added to verify the correctness of the resized image.
void writeResizedImage(const std::string &outputFilename, const std::vector<std::vector<std::vector<int>>> &inputMatrix)
{
  // Get dimensions of the resized image
  int depth = inputMatrix.size();       // Should be 3 (R, G, B)
  int height = inputMatrix[0].size();   // Should be 224
  int width = inputMatrix[0][0].size(); // Should be 224

  // Create a JPEG file pointer
  FILE *outfile = fopen(outputFilename.c_str(), "wb");
  if (!outfile)
  {
    std::cerr << "Error opening output JPEG file: " << outputFilename << std::endl;
    return;
  }

  // Initialize the JPEG compression object
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  // Specify the output file
  jpeg_stdio_dest(&cinfo, outfile);

  // Set image parameters
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = depth; // Number of color components (RGB)
  cinfo.in_color_space = JCS_RGB; // Colorspace of input image

  // Set default compression parameters
  jpeg_set_defaults(&cinfo);
  jpeg_set_quality(&cinfo, 90, TRUE); // Adjust quality settings here (optional)

  // Start compression
  jpeg_start_compress(&cinfo, TRUE);

  // Write scanlines of image data
  JSAMPROW row_pointer = new JSAMPLE[width * depth]; // Allocate memory for one row

  while (cinfo.next_scanline < cinfo.image_height)
  {
    // Fill row_pointer with RGB pixel values from inputMatrix
    for (int w = 0; w < width; ++w)
    {
      for (int d = 0; d < depth; ++d)
      {
        row_pointer[w * depth + d] = (JSAMPLE)inputMatrix[d][cinfo.next_scanline][w];
      }
    }
    // Write scanline to the JPEG file
    jpeg_write_scanlines(&cinfo, &row_pointer, 1);
  }

  // Finish compression
  jpeg_finish_compress(&cinfo);

  // Clean up
  jpeg_destroy_compress(&cinfo);
  fclose(outfile);

  // Deallocate the row pointer
  delete[] row_pointer;

  std::cout << "Resized image saved as JPEG: " << outputFilename << std::endl;
}
#endif // ifdef COMPILE_WITH_JPEG

#endif