// File: pimPerfEnergyFulcrum.h
// PIMeval Simulator - Performance Energy Models
// Copyright (c) 2024 University of Virginia
// This file is licensed under the MIT License.
// See the LICENSE file in the root of this repository for more details.

#ifndef LAVA_PIM_PERF_ENERGY_FULCRUM_H
#define LAVA_PIM_PERF_ENERGY_FULCRUM_H

#include "libpimeval.h"        // for PimDeviceEnum, PimDataType
#include "pimParamsDram.h"     // for pimParamsDram
#include "pimCmd.h"            // for PimCmdEnum
#include "pimResMgr.h"         // for pimObjInfo
#include "pimPerfEnergyBase.h" // for pimPerfEnergyBase
#include <algorithm>
#include <cstdlib>

//! @class  pimPerfEnergyBitFulcrum
//! @brief  PIM performance energy model for Fulcrum family
class pimPerfEnergyFulcrum : public pimPerfEnergyBase
{
public:
  pimPerfEnergyFulcrum(const pimPerfEnergyModelParams &params) : pimPerfEnergyBase(params)
  {
    // Sweep knobs without recompiling.
    //   ONCNN_PES_PER_CORE : k, ON-CNN PEs per PIM core            (default 4)
    //   ONCNN_DELTA_M      : online delay of the multiplier        (3 = ON-CNN_SS, 2 = ON-CNN_SP)
    //   ONCNN_PA           : activation precision Pa in bits       (default 16, as in the paper)
    //   ONCNN_CLK_NS       : PE clock period in ns                 (1.2 = SS, 1.1 = SP, paper Sec. IV-C)
    //   ONCNN_PORT_BITS    : walker->PE bits/cycle, 0 = ideal      (default 0, paper has no such limit)
    //   ONCNN_ROW_ACCESS   : 1 = also charge DRAM row rd/wr time   (default 0, Eq. 10 has no such term)
    auto envLong = [](const char *name, long def, long minV) -> long {
      const char *s = std::getenv(name);
      if (!s || !*s) return def;
      return std::max(minV, std::atol(s));
    };
    m_oncnnPEsPerCore = (unsigned)envLong("ONCNN_PES_PER_CORE", m_oncnnPEsPerCore, 1);
    m_oncnnDeltaM = (unsigned)envLong("ONCNN_DELTA_M", m_oncnnDeltaM, 0);
    m_oncnnPa = (unsigned)envLong("ONCNN_PA", m_oncnnPa, 1);
    m_oncnnPortBits = (unsigned)envLong("ONCNN_PORT_BITS", m_oncnnPortBits, 0);
    m_oncnnChargeRowAccess = envLong("ONCNN_ROW_ACCESS", 0, 0) != 0;
    if (const char *s = std::getenv("ONCNN_CLK_NS"))
      if (std::atof(s) > 0.0)
        m_oncnnClkMs = std::atof(s) * 1e-6; // ns -> ms
  }
  virtual ~pimPerfEnergyFulcrum() {}

  virtual pimeval::perfEnergy getPerfEnergyForFunc1(PimCmdEnum cmdType, const pimObjInfo &objSrc, const pimObjInfo &objDest) const override;
  virtual pimeval::perfEnergy getPerfEnergyForFunc2(PimCmdEnum cmdType, const pimObjInfo &objSrc1, const pimObjInfo &objSrc2, const pimObjInfo &objDest) const override;
  virtual pimeval::perfEnergy getPerfEnergyForReduction(PimCmdEnum cmdType, const pimObjInfo &obj, unsigned numPass) const override;
  virtual pimeval::perfEnergy getPerfEnergyForBroadcast(PimCmdEnum cmdType, const pimObjInfo &obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForRotate(PimCmdEnum cmdType, const pimObjInfo &obj) const override;
  virtual pimeval::perfEnergy getPerfEnergyForPrefixSum(PimCmdEnum cmdType, const pimObjInfo &obj) const override;

protected:
  double m_fulcrumMulLatency = 0.00000609; // 6.09ns
  double m_fulcrumAddLatency = 0.00000120; // 1.20ns
  unsigned m_fulcrumAluBitWidth = 32;
  // Following values are taken from fulcrum paper.
  double m_fulcrumMulEnergy = 0.0000000004992329586; // mJ
  double m_fulcrumAddEnergy = 0.0000000001467846411; // mJ
  double m_fulcrumShiftEnergy = 0.0000000075;        // mJ

  // ON-CNN PE block (Shafique & Lee, IEEE Access 2024)
  double m_oncnnClkMs = 1.2e-6;         // 1.2 ns (ON-CNN_SS synthesis result, Eq. 11); 1.1 ns for SP
  double m_oncnnGroupEnergyMj = 1.9e-7; // ~190 pJ per 16-SOP run (paper Table 6 power / 4096 PEs)
  unsigned m_oncnnDeltaM = 3;           // delta_M: 3 (OSSM) or 2 (OSPM)
  unsigned m_oncnnDeltaA = 2;           // delta_A: online adder delay
  unsigned m_oncnnPa = 16;              // Pa: activation precision in bits (paper: 16)
  unsigned m_oncnnPEsPerCore = 4;       // k
  unsigned m_oncnnPortBits = 0;         // 0 = ideal (paper); >0 limits walker->PE bits per cycle
  bool m_oncnnChargeRowAccess = true;  // false = paper-exact Eq. 10 (no DRAM row time)
  bool m_oncnnChargeDestWrite = true;   // only used when m_oncnnChargeRowAccess / energy
};

#endif