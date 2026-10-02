// File: libpimeval.cpp
// PIMeval Simulator - Library Interface
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#include "libpimeval.h"
#include "pimSim.h"
#include "pimUtils.h"

//! @brief  Create a PIM device
PimStatus
pimCreateDevice(PimDeviceEnum deviceType, unsigned numRanks, unsigned numBankPerRank, unsigned numSubarrayPerBank, unsigned numRows, unsigned numCols, unsigned bufferSize)
{
  bool ok = pimSim::get()->createDevice(deviceType, numRanks, numBankPerRank, numSubarrayPerBank, numRows, numCols, bufferSize);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Create a PIM device from config file
PimStatus
pimCreateDeviceFromConfig(PimDeviceEnum deviceType, const char *configFileName)
{
  bool ok = pimSim::get()->createDeviceFromConfig(deviceType, configFileName);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Get PIM device properties
PimStatus
pimGetDeviceProperties(PimDeviceProperties *deviceProperties)
{
  bool ok = pimSim::get()->getDeviceProperties(deviceProperties);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Delete a PIM device
PimStatus
pimDeleteDevice()
{
  bool ok = pimSim::get()->deleteDevice();
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus
pimPrefixSum(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimPrefixSum(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Start timer for a PIM kernel to measure CPU runtime and DRAM refresh
void pimStartTimer()
{
  pimSim::get()->startKernelTimer();
}

//! @brief  End timer for a PIM kernel to measure CPU runtime and DRAM refresh
void pimEndTimer()
{
  pimSim::get()->endKernelTimer();
}

//! @brief  Show PIM command stats
void pimShowStats()
{
  pimSim::get()->showStats();
}

//! @brief  Reset PIM command stats
void pimResetStats()
{
  pimSim::get()->resetStats();
}

//! @brief  Is analysis mode. Call this after device creation
bool pimIsAnalysisMode()
{
  return pimSim::get()->isAnalysisMode();
}

//! @brief  Allocate a PIM resource
PimObjId
pimAlloc(PimAllocEnum allocType, uint64_t numElements, PimDataType dataType)
{
  return pimSim::get()->pimAlloc(allocType, numElements, dataType);
}

//! @brief  Allocate a PIM resource, with an associated object as reference
PimObjId
pimAllocAssociated(PimObjId assocId, PimDataType dataType)
{
  return pimSim::get()->pimAllocAssociated(assocId, dataType);
}

//! @brief  Allocate a global buffer for broadcasting data to all PIM cores
PimObjId
pimAllocBuffer(uint32_t numElements, PimDataType dataType)
{
  return pimSim::get()->pimAllocBuffer(numElements, dataType);
}

//! @brief  Free a PIM resource
PimStatus
pimFree(PimObjId obj)
{
  bool ok = pimSim::get()->pimFree(obj);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Create an obj referencing to a range of an existing obj
PimObjId
pimCreateRangedRef(PimObjId refId, uint64_t idxBegin, uint64_t idxEnd)
{
  return pimSim::get()->pimCreateRangedRef(refId, idxBegin, idxEnd);
}

//! @brief  Create an obj referencing to negation of an existing obj based on dual-contact memory cells
PimObjId
pimCreateDualContactRef(PimObjId refId)
{
  return pimSim::get()->pimCreateDualContactRef(refId);
}

//! @brief  Copy data from main memory to PIM device for a range of elements within the PIM object
PimStatus
pimCopyHostToDevice(void *src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyMainToDevice(src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to main memory for a range of elements within the PIM object
PimStatus
pimCopyDeviceToHost(PimObjId src, void *dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyDeviceToMain(src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from main memory to PIM device with type for a range of elements within the PIM object
PimStatus
pimCopyHostToDeviceWithType(PimCopyEnum copyType, void *src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyMainToDeviceWithType(copyType, src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to main memory with type for a range of elements within the PIM object
PimStatus
pimCopyDeviceToHostWithType(PimCopyEnum copyType, PimObjId src, void *dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyDeviceToMainWithType(copyType, src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Copy data from PIM device to device for a range of elements within the PIM object
PimStatus
pimCopyDeviceToDevice(PimObjId src, PimObjId dest, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimCopyDeviceToDevice(src, dest, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimCopyObjectToObject(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimCopyObjectToObject(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Convert data type between two associated PIM objects of different data types
PimStatus pimConvertType(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimConvertType(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Load vector with a signed int value
PimStatus
pimBroadcastInt(PimObjId dest, int64_t value)
{
  bool ok = pimSim::get()->pimBroadcast(dest, value);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Load vector with an unsigned int value
PimStatus
pimBroadcastUInt(PimObjId dest, uint64_t value)
{
  bool ok = pimSim::get()->pimBroadcast(dest, value);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Load vector with a float32 value
PimStatus
pimBroadcastFP(PimObjId dest, float value)
{
  bool ok = pimSim::get()->pimBroadcast(dest, value);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM add
PimStatus
pimAdd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimAdd(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM sub
PimStatus
pimSub(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimSub(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM div
PimStatus
pimDiv(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimDiv(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM not
PimStatus
pimNot(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimNot(src, dest);
  ;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM or
PimStatus
pimOr(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimOr(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM and
PimStatus
pimAnd(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimAnd(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM xor
PimStatus
pimXor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimXor(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM xnor
PimStatus
pimXnor(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimXnor(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM abs
PimStatus
pimAbs(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimAbs(src, dest);
  ;
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM multiplication
PimStatus
pimMul(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMul(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM GT
PimStatus
pimGT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimGT(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM LT
PimStatus
pimLT(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimLT(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM EQ
PimStatus
pimEQ(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimEQ(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM NE
PimStatus
pimNE(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimNE(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Min
PimStatus
pimMin(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMin(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Max
PimStatus
pimMax(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimMax(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimAddScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimAdd(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimSubScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimSub(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimMulScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimMul(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimDivScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimDiv(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimAndScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimAnd(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimOrScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimOr(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimXorScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimXor(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimXnorScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimXnor(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimGTScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimGT(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimLTScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimLT(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimEQScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimEQ(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimNEScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimNE(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimMinScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimMin(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimMaxScalar(PimObjId src, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimMax(src, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimScaledAdd(PimObjId src1, PimObjId src2, PimObjId dest, uint64_t scalarValue)
{
  bool ok = pimSim::get()->pimScaledAdd(src1, src2, dest, scalarValue);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM Pop Count
PimStatus
pimPopCount(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimPopCount(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Extract a bit slice from a data vector. Dest must be BOOL type
PimStatus
pimBitSliceExtract(PimObjId src, PimObjId destBool, unsigned bitIdx)
{
  bool ok = pimSim::get()->pimBitSliceExtract(src, destBool, bitIdx);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Insert a bit slice to a data vector. Src must be BOOL type
PimStatus
pimBitSliceInsert(PimObjId srcBool, PimObjId dest, unsigned bitIdx)
{
  bool ok = pimSim::get()->pimBitSliceInsert(srcBool, dest, bitIdx);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional copy: dest[i] = cond ? src[i] : dest[i]
PimStatus
pimCondCopy(PimObjId condBool, PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimCondCopy(condBool, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional broadcast: dest[i] = cond ? scalar : dest[i]
PimStatus
pimCondBroadcast(PimObjId condBool, uint64_t scalarBits, PimObjId dest)
{
  bool ok = pimSim::get()->pimCondBroadcast(condBool, scalarBits, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional select: dest[i] = cond ? src1[i] : src2[i]
PimStatus
pimCondSelect(PimObjId condBool, PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimCondSelect(condBool, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Conditional select scalar: dest[i] = cond ? src1[i] : scalar
PimStatus
pimCondSelectScalar(PimObjId condBool, PimObjId src1, uint64_t scalarBits, PimObjId dest)
{
  bool ok = pimSim::get()->pimCondSelectScalar(condBool, src1, scalarBits, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  AES Sbox: dest[i] = lut[src[i]]
PimStatus
pimAesSbox(PimObjId src, PimObjId dest, const std::vector<uint8_t> &lut)
{
  bool ok = pimSim::get()->pimAesSbox(src, dest, lut);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  AES Sbox: dest[i] = lut[src[i]] (similar to AES sbox, different in perforamance and energy model for the bit-serial architecture)
PimStatus
pimAesInverseSbox(PimObjId src, PimObjId dest, const std::vector<uint8_t> &lut)
{
  bool ok = pimSim::get()->pimAesInverseSbox(src, dest, lut);
  return ok ? PIM_OK : PIM_ERROR;
}

// Implementation of min reduction
PimStatus pimRedMin(PimObjId src, void *min, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimRedMin(src, min, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

// Implementation of max reduction
PimStatus pimRedMax(PimObjId src, void *max, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimRedMax(src, max, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM MAC operation: dest += src1 * src2
PimStatus pimMAC(PimObjId src1, PimObjId src2, void *dest)
{
  bool ok = pimSim::get()->pimMAC(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  PIM reduction sum for signed int. Result returned to a host variable
PimStatus
pimRedSum(PimObjId src, void *sum, uint64_t idxBegin, uint64_t idxEnd)
{
  bool ok = pimSim::get()->pimRedSum(src, sum, idxBegin, idxEnd);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Rotate all elements of an obj by one step to the right
PimStatus
pimRotateElementsRight(PimObjId src)
{
  bool ok = pimSim::get()->pimRotateElementsRight(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Rotate all elements of an obj by one step to the left
PimStatus
pimRotateElementsLeft(PimObjId src)
{
  bool ok = pimSim::get()->pimRotateElementsLeft(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift elements of an obj by one step to the right and fill zero
PimStatus
pimShiftElementsRight(PimObjId src)
{
  bool ok = pimSim::get()->pimShiftElementsRight(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift elements of an obj by one step to the left and fill zero
PimStatus
pimShiftElementsLeft(PimObjId src)
{
  bool ok = pimSim::get()->pimShiftElementsLeft(src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift bits of each elements of an obj by shiftAmount to the right. This currently implements arithmetic shift.
PimStatus
pimShiftBitsRight(PimObjId src, PimObjId dest, unsigned shiftAmount)
{
  bool ok = pimSim::get()->pimShiftBitsRight(src, dest, shiftAmount);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Shift bits of each elements of an obj by shiftAmount to the left.
PimStatus
pimShiftBitsLeft(PimObjId src, PimObjId dest, unsigned shiftAmount)
{
  bool ok = pimSim::get()->pimShiftBitsLeft(src, dest, shiftAmount);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  Execute fused PIM APIs
PimStatus
pimFuse(PimProg prog)
{
  bool ok = pimSim::get()->pimFuse(prog);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Read a row to SA
PimStatus
pimOpReadRowToSa(PimObjId src, unsigned ofst)
{
  bool ok = pimSim::get()->pimOpReadRowToSa(src, ofst);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Write SA to a row
PimStatus
pimOpWriteSaToRow(PimObjId src, unsigned ofst)
{
  bool ok = pimSim::get()->pimOpWriteSaToRow(src, ofst);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Triple row activation to SA
PimStatus
pimOpTRA(PimObjId src1, unsigned ofst1, PimObjId src2, unsigned ofst2, PimObjId src3, unsigned ofst3)
{
  bool ok = pimSim::get()->pimOpTRA(src1, ofst1, src2, ofst2, src3, ofst3);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Move value between two regs
PimStatus
pimOpMove(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpMove(objId, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Set value of a reg
PimStatus
pimOpSet(PimObjId objId, PimRowReg src, bool val)
{
  bool ok = pimSim::get()->pimOpSet(objId, src, val);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Not of a reg
PimStatus
pimOpNot(PimObjId objId, PimRowReg src, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNot(objId, src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: And of two regs
PimStatus
pimOpAnd(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpAnd(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Or of two regs
PimStatus
pimOpOr(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpOr(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Nand of two regs
PimStatus
pimOpNand(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNand(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Nor of two regs
PimStatus
pimOpNor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpNor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Xor of two regs
PimStatus
pimOpXor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpXor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Xnor of two regs
PimStatus
pimOpXnor(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpXnor(objId, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Maj of three regs
PimStatus
pimOpMaj(PimObjId objId, PimRowReg src1, PimRowReg src2, PimRowReg src3, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpMaj(objId, src1, src2, src3, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Conditional selecion: dest = cond ? src1 : src2
PimStatus
pimOpSel(PimObjId objId, PimRowReg cond, PimRowReg src1, PimRowReg src2, PimRowReg dest)
{
  bool ok = pimSim::get()->pimOpSel(objId, cond, src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Rotate a reg to the right, using srcId for range
PimStatus
pimOpRotateRH(PimObjId objId, PimRowReg src)
{
  bool ok = pimSim::get()->pimOpRotateRH(objId, src);
  return ok ? PIM_OK : PIM_ERROR;
}

//! @brief  BitSIMD-V: Rotate a reg to the left, using srcId for range
PimStatus
pimOpRotateLH(PimObjId objId, PimRowReg src)
{
  bool ok = pimSim::get()->pimOpRotateLH(objId, src);
  return ok ? PIM_OK : PIM_ERROR;
}

// @brief  SIMDRAM: AP operation
PimStatus
pimOpAP(int numSrc, ...)
{
  va_list args;
  va_start(args, numSrc);
  bool ok = pimSim::get()->pimOpAP(numSrc, args);
  va_end(args);
  return ok ? PIM_OK : PIM_ERROR;
}

// @brief  SIMDRAM: AAP operation
PimStatus
pimOpAAP(int numSrc, int numDest, ...)
{
  va_list args;
  va_start(args, numDest);
  bool ok = pimSim::get()->pimOpAAP(numSrc, numDest, args);
  va_end(args);
  return ok ? PIM_OK : PIM_ERROR;
}

// =====================================================================
// ON-CNN Specific High-Level PIM Implementations
// =====================================================================
// below functions are micro level implementation --> takes a lot of time to run

// PimStatus pimOSSM(PimObjId srcX, PimObjId srcY, PimObjId destP, int numBits)
// {
//   PimObjId W_sum = pimAllocAssociated(srcX, PIM_INT32);
//   pimBroadcastInt(W_sum, 0);
//   PimObjId W_carry = pimAllocAssociated(srcX, PIM_INT32);
//   pimBroadcastInt(W_carry, 0);
//   PimObjId X_reg = pimAllocAssociated(srcX, PIM_INT32);
//   pimBroadcastInt(X_reg, 0);
//   PimObjId Y_reg = pimAllocAssociated(srcY, PIM_INT32);
//   pimBroadcastInt(Y_reg, 0);

//   PimObjId x_j_bool = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId y_j_bool = pimAllocAssociated(srcY, PIM_BOOL);
//   PimObjId partial_X = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId partial_Y = pimAllocAssociated(srcY, PIM_INT32);

//   PimObjId v_est_sum = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId v_est_carry = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId S1 = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId C1 = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId C1_shifted = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId C2 = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId temp1 = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId temp2 = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId temp3 = pimAllocAssociated(srcX, PIM_INT32);

//   // Temps for SELM & M-Block
//   PimObjId v_S_top = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId v_C_top = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId v_top = pimAllocAssociated(srcX, PIM_INT32);
//   PimObjId v_m1 = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId v_0 = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId v_1 = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId v_2 = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId pp = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId pn = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId p_mag = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId not_vm1 = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId not_v0 = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId not_v1 = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId temp_or = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId v_0_star = pimAllocAssociated(srcX, PIM_BOOL);
//   PimObjId mask_29 = pimAllocAssociated(srcX, PIM_INT32);
//   pimBroadcastInt(mask_29, 0x1FFFFFFF);

//   // OFC Init
//   PimObjId destQM = pimAllocAssociated(srcX, PIM_INT32);
//   pimBroadcastInt(destP, 0);
//   pimBroadcastInt(destQM, 0);

//   for (int j = -3; j < numBits; ++j)
//   {
//     int k = j + 4;
//     if (k >= 1 && k <= numBits)
//     {
//       pimBitSliceExtract(srcX, x_j_bool, numBits - k);
//       pimBitSliceExtract(srcY, y_j_bool, numBits - k);
//       // ALIGNMENT FIX: Position relative to fractional point
//       pimBitSliceInsert(x_j_bool, X_reg, 30 - k);
//       pimBitSliceInsert(y_j_bool, Y_reg, 30 - k);
//     }
//     else
//     {
//       pimBroadcastInt(x_j_bool, 0);
//       pimBroadcastInt(y_j_bool, 0);
//     }

//     pimShiftBitsLeft(W_sum, W_sum, 1);
//     pimShiftBitsLeft(W_carry, W_carry, 1);

//     pimCondBroadcast(y_j_bool, 0xFFFFFFFF, partial_X);
//     pimAnd(X_reg, partial_X, partial_X);
//     pimCondBroadcast(x_j_bool, 0xFFFFFFFF, partial_Y);
//     pimAnd(Y_reg, partial_Y, partial_Y);

//     pimShiftBitsRight(partial_X, partial_X, 3);
//     pimShiftBitsRight(partial_Y, partial_Y, 3);

//     // --- 4:2 Compressor ---
//     pimXor(partial_X, partial_Y, temp1);
//     pimXor(temp1, W_sum, S1);
//     pimAnd(partial_X, partial_Y, temp2);
//     pimAnd(W_sum, temp1, temp3);
//     pimOr(temp2, temp3, C1);
//     pimShiftBitsLeft(C1, C1_shifted, 1);

//     pimXor(S1, C1_shifted, temp1);
//     pimXor(temp1, W_carry, v_est_sum);
//     pimAnd(S1, C1_shifted, temp2);
//     pimAnd(W_carry, temp1, temp3);
//     pimOr(temp2, temp3, C2);
//     pimShiftBitsLeft(C2, v_est_carry, 1);

//     // --- SELM (Digit Selection) ---
//     pimShiftBitsRight(v_est_sum, v_S_top, 28);
//     pimShiftBitsRight(v_est_carry, v_C_top, 28);
//     pimAdd(v_S_top, v_C_top, v_top); // Assimilate top 4 bits

//     pimBitSliceExtract(v_top, v_m1, 3);
//     pimBitSliceExtract(v_top, v_0, 2);
//     pimBitSliceExtract(v_top, v_1, 1);
//     pimBitSliceExtract(v_top, v_2, 0);

//     pimNot(v_m1, not_vm1);
//     pimOr(v_0, v_1, temp_or);
//     pimAnd(not_vm1, temp_or, pp); // pp = 1

//     pimNot(v_0, not_v0);
//     pimNot(v_1, not_v1);
//     pimOr(not_v0, not_v1, temp_or);
//     pimAnd(v_m1, temp_or, pn); // pn = -1

//     pimOr(pp, pn, p_mag); // p_mag = |p|

//     // --- M-Block ---
//     pimXor(v_0, p_mag, v_0_star);
//     pimAnd(v_est_sum, mask_29, W_sum);
//     pimAnd(v_est_carry, mask_29, W_carry);

//     pimBitSliceInsert(v_0_star, W_sum, 31);
//     pimBitSliceInsert(v_1, W_sum, 30);
//     pimBitSliceInsert(v_2, W_sum, 29);

//     // --- OFC (In-Place Generation) ---
//     if (j >= 0)
//     {
//       pimOFC(p_mag, pn, destP, destQM, numBits - 1 - j);
//     }
//   }

//   pimFree(W_sum);
//   pimFree(W_carry);
//   pimFree(X_reg);
//   pimFree(Y_reg);
//   pimFree(x_j_bool);
//   pimFree(y_j_bool);
//   pimFree(partial_X);
//   pimFree(partial_Y);
//   pimFree(v_est_sum);
//   pimFree(v_est_carry);
//   pimFree(S1);
//   pimFree(C1);
//   pimFree(C1_shifted);
//   pimFree(C2);
//   pimFree(temp1);
//   pimFree(temp2);
//   pimFree(temp3);
//   pimFree(v_S_top);
//   pimFree(v_C_top);
//   pimFree(v_top);
//   pimFree(v_m1);
//   pimFree(v_0);
//   pimFree(v_1);
//   pimFree(v_2);
//   pimFree(pp);
//   pimFree(pn);
//   pimFree(p_mag);
//   pimFree(not_vm1);
//   pimFree(not_v0);
//   pimFree(not_v1);
//   pimFree(temp_or);
//   pimFree(v_0_star);
//   pimFree(mask_29);
//   pimFree(destQM);

//   return PIM_OK;
// }

// PimStatus pimOFC(PimObjId q_mag, PimObjId q_sign, PimObjId destQ, PimObjId destQM, int step)
// {
//   // Step 1: Identify conditions based on magnitude and sign of the Signed-Digit
//   PimObjId q_is_1 = pimAllocAssociated(q_mag, PIM_BOOL);
//   PimObjId q_is_minus_1 = pimAllocAssociated(q_mag, PIM_BOOL);
//   PimObjId not_sign = pimAllocAssociated(q_sign, PIM_BOOL);
//   PimObjId not_mag = pimAllocAssociated(q_mag, PIM_BOOL);

//   // Condition 1: q == +1 -> Magnitude is 1 AND Sign is 0
//   pimNot(q_sign, not_sign);
//   pimAnd(q_mag, not_sign, q_is_1);

//   // Condition 2: q == -1 -> Magnitude is 1 AND Sign is 1
//   pimAnd(q_mag, q_sign, q_is_minus_1);

//   // Step 2: Parallel In-Memory Conditional Swapping (Using pimCondCopy)
//   // This perfectly emulates the OFC logic without moving data to the CPU

//   // If q == 1: QM[0:step-1] = Q[0:step-1]
//   // pimCondCopy writes data from Source to Dest ONLY where condition vector is TRUE
//   pimCondCopy(q_is_1, destQ, destQM);

//   // If q == -1: Q[0:step-1] = QM[0:step-1]
//   pimCondCopy(q_is_minus_1, destQM, destQ);

//   // Step 3: Insert the current bit at the specific 'step' position
//   // Q[step] gets 1 if magnitude is 1 (q = 1 or -1)
//   pimBitSliceInsert(q_mag, destQ, step);

//   // QM[step] gets 1 if magnitude is 0 (q = 0)
//   pimNot(q_mag, not_mag);
//   pimBitSliceInsert(not_mag, destQM, step);

//   // Cleanup
//   pimFree(q_is_1);
//   pimFree(q_is_minus_1);
//   pimFree(not_sign);
//   pimFree(not_mag);

//   return PIM_OK;
// }

// PimStatus pimOATreeReduce(PimObjId srcP, PimObjId destSum)
// {
//   // The ON-CNN PE requires summing 16 multipliers in an Adder Tree.
//   // In a bit-serial PIM architecture, this means reducing 16 adjacent elements.
//   // We achieve this using a $\log_2(16) = 4$ level in-memory reduction tree
//   // by utilizing element shifting and parallel addition.

//   PimObjId temp = pimAllocAssociated(srcP, PIM_INT32);

//   // Initialize destination with source values
//   pimCopyObjectToObject(srcP, destSum);

//   // 4-level reduction tree (1, 2, 4, 8 shifts)
//   int shift_amounts[4] = {1, 2, 4, 8};

//   for (int i = 0; i < 4; i++)
//   {
//     // Copy current state to temp
//     pimCopyObjectToObject(destSum, temp);

//     // Shift 'temp' elements to the right to align the adjacent pairs
//     for (int s = 0; s < shift_amounts[i]; s++)
//     {
//       // This API shifts the entire vector across the memory array boundary
//       pimShiftElementsRight(temp);
//     }

//     // Perform parallel in-memory addition: destSum = destSum + temp
//     pimAdd(destSum, temp, destSum);
//   }

//   pimFree(temp);
//   return PIM_OK;
// }

// Implementation of ON-CNN APIs
PimStatus pimOSSM(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimOSSM(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimOFC(PimObjId src, PimObjId dest)
{
  bool ok = pimSim::get()->pimOFC(src, dest);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimOAReduce(PimObjId src, void *result)
{
  bool ok = pimSim::get()->pimOAReduce(src, result);
  return ok ? PIM_OK : PIM_ERROR;
}

PimStatus pimONCNNMac(PimObjId src1, PimObjId src2, PimObjId dest)
{
  bool ok = pimSim::get()->pimONCNNMac(src1, src2, dest);
  return ok ? PIM_OK : PIM_ERROR;
}
