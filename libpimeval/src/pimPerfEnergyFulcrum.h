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
#include <cstring>

//! @class  pimPerfEnergyFulcrum
//! @brief  PIM performance energy model for Fulcrum family
//!
//! ON-CNN mapping notes (Shafique & Lee, IEEE Access 2024):
//!   - Variant knob (ONCNN_VARIANT=SS|SP, default SS): selects online multiplier delay
//!     delta_M (3 for OSSM in SS, 2 for OSPM in SP) and clock period (1.2 ns for SS,
//!     1.1 ns for SP), deriving C_SOP from Eq. 7:
//!       C_SOP = delta_M + delta_A * ceil(log2(M)) + 2 * Pa + ceil(log2(M))
//!     which yields 47 cycles (SS) or 46 cycles (SP) for M=16, delta_A=2, Pa=16.
//!   - Intentionally NOT modeled in this perf/energy class or handled host-side:
//!     1) Requantization to int16 is performed on the host per layer (dynamic percentile shift).
//!     2) ReLU and MaxPooling are executed outside conv2ONCNN rather than on the paper's PPU..
//!     3) Tile and column power gating (Sec. IV-D, Tables 3-4) are not subtracted in the
//!        energy model; all configured cores in the scaled RAM config are charged.
class pimPerfEnergyFulcrum : public pimPerfEnergyBase
{
public:
  pimPerfEnergyFulcrum(const pimPerfEnergyModelParams &params) : pimPerfEnergyBase(params)
  {
    // Sweep & paper-alignment knobs (no recompile needed):
    //   ONCNN_VARIANT      : SS (default: dM=3, clk=1.2ns, C_SOP=47) or SP (dM=2, clk=1.1ns, C_SOP=46)
    //   ONCNN_PES_PER_CORE : k, ON-CNN PEs per PIM core                     (default 4)
    //   ONCNN_DELTA_M      : override online delay of multiplier            (default from ONCNN_VARIANT)
    //   ONCNN_PA           : activation precision Pa in bits                (default 16, paper Sec. IV-B)
    //   ONCNN_CLK_NS       : override PE clock period in ns                 (default from ONCNN_VARIANT)
    //   ONCNN_PORT_BITS    : walker->PE bits/cycle; 0 = ideal unthrottled   (default 128)
    //   ONCNN_ROW_ACCESS   : 1 = charge DRAM row rd/wr latency, 0 = Eq. 10  (default 1 / true)
    auto envLong = [](const char *name, long def, long minV) -> long
    {
      const char *s = std::getenv(name);
      if (!s || !*s)
        return def;
      return std::max(minV, std::atol(s));
    };

    const char *varEnv = std::getenv("ONCNN_VARIANT");
    const bool isSP = (varEnv && (std::strcmp(varEnv, "SP") == 0 || std::strcmp(varEnv, "sp") == 0));
    m_oncnnDeltaM = isSP ? 2u : 3u;
    m_oncnnClkMs = (isSP ? 1.1 : 1.2) * 1e-6; // ns -> ms

    m_oncnnPEsPerCore = (unsigned)envLong("ONCNN_PES_PER_CORE", m_oncnnPEsPerCore, 1);
    m_oncnnDeltaM = (unsigned)envLong("ONCNN_DELTA_M", m_oncnnDeltaM, 0);
    m_oncnnPa = (unsigned)envLong("ONCNN_PA", m_oncnnPa, 1);
    m_oncnnPortBits = (unsigned)envLong("ONCNN_PORT_BITS", m_oncnnPortBits, 0);
    m_oncnnChargeRowAccess = envLong("ONCNN_ROW_ACCESS", m_oncnnChargeRowAccess ? 1 : 0, 0) != 0;
    if (const char *s = std::getenv("ONCNN_CLK_NS"))
      if (std::atof(s) > 0.0)
        m_oncnnClkMs = std::atof(s) * 1e-6; // ns -> ms
    const char *pc = std::getenv("ONCNN_PORT_CLK_NS");
    m_oncnnPortClkMs = (pc && *pc) ? std::atof(pc) * 1e-6 : m_oncnnClkMs; // default = PE clock
  }
  virtual ~pimPerfEnergyFulcrum() {}

  // Derive C_SOP from Paper Eq. 7: C_SOP = dM + dA * ceil(log2(M)) + 2 * Pa + ceil(log2(M))
  unsigned getOnCnnCsop(unsigned lanes = 16) const
  {
    unsigned treeLvl = 0;
    for (unsigned v = 1; v < lanes; v <<= 1)
      ++treeLvl;
    return m_oncnnDeltaM + m_oncnnDeltaA * treeLvl + 2 * m_oncnnPa + treeLvl;
  }

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
  double m_oncnnClkMs = 1.2e-6;         // 1.2 ns (ON-CNN_SS default, Eq. 11); 1.1 ns for ON-CNN_SP
  double m_oncnnPortClkMs;              // period of the walker->PE port, in ms
  double m_oncnnGroupEnergyMj = 1.9e-7; // ~190 pJ per 16-SOP run (paper Table 6 power / 4096 PEs)
  unsigned m_oncnnDeltaM = 3;           // delta_M: 3 (OSSM / SS default) or 2 (OSPM / SP)
  unsigned m_oncnnDeltaA = 2;           // delta_A: online adder delay (2 cycles, Sec. III-A3)
  unsigned m_oncnnPa = 16;              // Pa: activation precision in bits (paper: 16)
  unsigned m_oncnnPEsPerCore = 4;       // k: ON-CNN PEs per Fulcrum core
  unsigned m_oncnnPortBits = 128;       // default 128 bits/cycle walker->PE port; 0 = ideal (paper)
  bool m_oncnnChargeRowAccess = true;   // default true (charge DRAM row rd/wr); false = paper Eq. 10
  bool m_oncnnChargeDestWrite = true;   // only used when m_oncnnChargeRowAccess / energy
};

#endif