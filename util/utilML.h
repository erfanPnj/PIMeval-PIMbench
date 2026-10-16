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

  int32_t redsum = 0;
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
constexpr int kOnCnnLanes = 16; // M: multipliers per PE
constexpr int kOnCnnCols = 16;  // C: IFM windows processed in parallel (paper, Sec. IV-A)

inline long oncnnEnvLong(const char *name, long def)
{
  const char *s = std::getenv(name);
  return (s && *s) ? std::atol(s) : def;
}
// Must match ONCNN_PES_PER_CORE used by the perf model (default 4)
inline int oncnnPEsPerCore() { return (int)std::max(1L, oncnnEnvLong("ONCNN_PES_PER_CORE", 4)); }
inline uint64_t oncnnCeilDiv(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

// Paper Eq. 7 with delta_A = 2, ceil(log2 M) = 4:  C_SOP = dM + 2*4 + 2*Pa + 4
inline uint64_t oncnnCsop()
{
  const long dM = std::max(0L, oncnnEnvLong("ONCNN_DELTA_M", 3));
  const long Pa = std::max(1L, oncnnEnvLong("ONCNN_PA", 16));
  return (uint64_t)(dM + 2 * 4 + 2 * Pa + 4);
}

// Paper Eq. 10 for a PE pool of numPEs PEs arranged as (numPEs/C) filters x C windows
inline uint64_t oncnnEq10Cycles(uint64_t Cout, uint64_t Hout, uint64_t Wout, uint64_t Cin,
                                uint64_t Hf, uint64_t Wf, uint64_t numPEs)
{
  const uint64_t cols = std::min<uint64_t>(kOnCnnCols, numPEs);
  const uint64_t rowsF = std::max<uint64_t>(1, numPEs / cols); // T x R
  return oncnnCeilDiv(Cout, rowsF) * oncnnCeilDiv(Hout * Wout, cols) *
         (oncnnCsop() + 1) * oncnnCeilDiv(Cin * Hf * Wf, kOnCnnLanes);
}

// One pass of the PE grid over one (filter block, window block).
// fillChunk(chunk, ifm, flt) fills the 16-lane operands of every PE for run number `chunk`.
// Each chunk is one pimONCNNMac call = one PE run (C_SOP + 1 cycles) on every PE.
bool performConvONCNN_Batched(const std::function<void(int, std::vector<int> &, std::vector<int> &)> &fillChunk,
                              int numChunks, int activePEs, std::vector<int> &outputMatrix)
{
  const int vectorLength = activePEs * kOnCnnLanes; // one 16-lane group per PE
  outputMatrix.assign(activePEs, 0);

  // IFM and filter operands: 16 lanes per PE, living side by side in the same regions.
  PimObjId ifmObject = pimAlloc(PIM_ALLOC_AUTO, vectorLength, PIM_INT32);
  if (ifmObject == -1)
  {
    std::cout << "Function: " << __func__ << " Abort: pimAlloc failed" << std::endl;
    return false;
  }
  PimObjId filterObject = pimAllocAssociated(ifmObject, PIM_INT32);
  // One accumulator element per PE: element p belongs to lanes [p*16, p*16+16) of the operand objects.
  PimObjId accObject = pimAllocAssociatedGrouped(ifmObject, PIM_INT32, kOnCnnLanes);
  if (filterObject == -1 || accObject == -1)
  {
    std::cout << "Function: " << __func__ << " Abort: associated allocation failed" << std::endl;
    if (filterObject != -1)
      pimFree(filterObject);
    if (accObject != -1)
      pimFree(accObject);
    pimFree(ifmObject);
    return false;
  }

  std::vector<int> ifmVec(vectorLength), fltVec(vectorLength), zeroInit(activePEs, 0);
  pimCopyHostToDevice((void *)zeroInit.data(), accObject);

  // Algorithm 3 innermost loop: ceil(Cin*Hf*Wf / M) runs per PE
  for (int chunk = 0; chunk < numChunks; ++chunk)
  {
    fillChunk(chunk, ifmVec, fltVec);
    pimCopyHostToDevice((void *)ifmVec.data(), ifmObject);
    pimCopyHostToDevice((void *)fltVec.data(), filterObject);
    pimONCNNMac(ifmObject, filterObject, accObject);
  }

  // activePEs ints come back, already reduced over the 16 lanes
  pimCopyDeviceToHost(accObject, outputMatrix.data());

  pimFree(accObject);
  pimFree(filterObject);
  pimFree(ifmObject);
  return true;
}

// Core ON-CNN convolution. weightAt(f, c, r, s) returns weight of filter f, channel c, row r, col s.
void conv2ONCNN(const std::vector<std::vector<std::vector<int>>> &inputMatrix,
                int kernelDepth, int kernelHeight, int kernelWidth,
                const std::function<int(int, int, int, int)> &weightAt,
                std::vector<std::vector<std::vector<int>>> &resultMatrix, int stride, int padding)
{
  PimDeviceProperties deviceProp;
  if (pimGetDeviceProperties(&deviceProp) != PIM_OK)
  {
    std::cout << "Abort: pimGetDeviceProperties failed" << std::endl;
    exit(1);
  }
  // 16-lane groups must not straddle a row region (32-bit elements)
  if (deviceProp.numColPerSubarray % (kOnCnnLanes * 32) != 0)
  {
    std::cout << "Abort: numColPerSubarray must be a multiple of 512 for ON-CNN 16-lane groups" << std::endl;
    exit(1);
  }

  const int inputDepth = inputMatrix.size(); // Cin
  const int inputHeight = inputMatrix[0].size();
  const int inputWidth = inputMatrix[0][0].size();
  const int outMatRow = (inputHeight + 2 * padding - kernelHeight) / stride + 1;
  const int outMatCol = (inputWidth + 2 * padding - kernelWidth) / stride + 1;
  const int totalWindows = outMatRow * outMatCol;            // H_out * W_out
  const int K = inputDepth * kernelHeight * kernelWidth;     // flattened reduction length
  const int numChunks = (K + kOnCnnLanes - 1) / kOnCnnLanes; // ceil(Cin*Hf*Wf / M)

  resultMatrix.assign(kernelDepth, std::vector<std::vector<int>>(outMatRow, std::vector<int>(outMatCol, 0)));

  // PE pool: k PEs per PIM core. Grid = (T*R filters) x (C windows), as in the paper.
  const uint64_t numPEs = (uint64_t)deviceProp.numPIMCores * oncnnPEsPerCore();
  const int cols = (int)std::min<uint64_t>(kOnCnnCols, numPEs); // C
  const int rowsF = (int)std::max<uint64_t>(1, numPEs / cols);  // T x R

  // Algorithm 3 / Eq. 6 loops: filter blocks, window blocks, chunks (inside performConvONCNN_Batched)
  for (int f0 = 0; f0 < kernelDepth; f0 += rowsF)
  {
    const int curF = std::min(rowsF, kernelDepth - f0);
    for (int w0 = 0; w0 < totalWindows; w0 += cols)
    {
      const int curW = std::min(cols, totalWindows - w0);
      const int activePEs = curF * curW; // PE index = filter * curW + window

      auto fill = [&](int chunk, std::vector<int> &ifm, std::vector<int> &flt)
      {
        // flat index kk = pixel * Cin + channel -> 16 lanes = 16 consecutive channels of a pixel
        int laneCh[kOnCnnLanes], laneKr[kOnCnnLanes], laneKc[kOnCnnLanes];
        int validLanes = 0;
        for (int l = 0; l < kOnCnnLanes; ++l)
        {
          const int kk = chunk * kOnCnnLanes + l;
          if (kk >= K)
            break;
          const int pix = kk / inputDepth;
          laneCh[l] = kk % inputDepth;
          laneKr[l] = pix / kernelWidth;
          laneKc[l] = pix % kernelWidth;
          ++validLanes;
        }
        std::fill(ifm.begin(), ifm.end(), 0); // unused lanes stay zero (idle multipliers)
        std::fill(flt.begin(), flt.end(), 0);
        for (int pe = 0; pe < activePEs; ++pe)
        {
          const int filt = f0 + pe / curW;
          const int win = w0 + pe % curW;
          const int oh = win / outMatCol, ow = win % outMatCol;
          for (int l = 0; l < validLanes; ++l)
          {
            const int idx = pe * kOnCnnLanes + l;
            const int ih = oh * stride + laneKr[l] - padding;
            const int iw = ow * stride + laneKc[l] - padding;
            flt[idx] = weightAt(filt, laneCh[l], laneKr[l], laneKc[l]);
            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
              ifm[idx] = inputMatrix[laneCh[l]][ih][iw];
          }
        }
      };

      std::vector<int> outVector;
      if (!performConvONCNN_Batched(fill, numChunks, activePEs, outVector))
        return;

      for (int pe = 0; pe < activePEs; ++pe)
      {
        const int filt = f0 + pe / curW;
        const int win = w0 + pe % curW;
        resultMatrix[filt][win / outMatCol][win % outMatCol] = outVector[pe];
      }
    }
  }

  if (std::getenv("ONCNN_PRINT_EQ10"))
  {
    const uint64_t cyc = oncnnEq10Cycles(kernelDepth, outMatRow, outMatCol, inputDepth,
                                         kernelHeight, kernelWidth, numPEs);
    const double clkNs = std::getenv("ONCNN_CLK_NS") ? std::atof(std::getenv("ONCNN_CLK_NS")) : 1.2;
    std::cout << "ON-CNN Eq.10 expected: " << cyc << " cycles = " << cyc * clkNs * 1e-6
              << " ms (PEs=" << numPEs << ", grid " << rowsF << "x" << cols << ")" << std::endl;
  }
}

// Existing signature (2D kernel per filter shared across channels, as the CSV loaders produce)
void conv2(std::vector<std::vector<std::vector<int>>> &inputMatrix, std::vector<std::vector<std::vector<int>>> &kernelMatrix, std::vector<std::vector<std::vector<int>>> &resultMatrix, int stride, int padding)
{
  conv2ONCNN(inputMatrix, kernelMatrix.size(), kernelMatrix[0].size(), kernelMatrix[0][0].size(), [&](int f, int, int r, int s)
             { return kernelMatrix[f][r][s]; }, resultMatrix, stride, padding);
}

// True 4D filters [Cout][Cin][Hf][Wf], as in the paper
void conv2(std::vector<std::vector<std::vector<int>>> &inputMatrix, std::vector<std::vector<std::vector<std::vector<int>>>> &kernel4D, std::vector<std::vector<std::vector<int>>> &resultMatrix, int stride, int padding)
{
  conv2ONCNN(inputMatrix, kernel4D.size(), kernel4D[0][0].size(), kernel4D[0][0][0].size(), [&](int f, int c, int r, int s)
             { return kernel4D[f][c][r][s]; }, resultMatrix, stride, padding);
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
