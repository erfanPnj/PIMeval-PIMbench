// Test: C++ version of ON-CNN Hardware Accelerator Benchmark
// Copyright (c) 2024
// This file is licensed under the MIT License.
//
// Description:
// This benchmark implements the application layer (Phase 4) for the ON-CNN architecture.
// It maps the Convolutional loops to the underlying Bit-Serial PIM primitives.
// Specifically, it utilizes the 16-multiplier PE design, the Online Adder (OA) Tree,
// and the On-the-Fly Converter (OFC) defined in the libpimeval core.

#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <getopt.h>
#include <iomanip>
#include <chrono>
#include <cassert>
#include "util.h"

using namespace std;

// =====================================================================
// Parameters Structure & Parser (Following vgg16.cpp standard)
// =====================================================================
typedef struct Params
{
    char *dramConfigFile;
    bool shouldVerify;
    bool moreDebugPrints;
    int numElements; // Number of parallel IFM windows to process
} Params;

void usage()
{
    fprintf(stderr,
            "\nUsage:  ./on_cnn.out [options]"
            "\n"
            "\n    -c    input file containing dramsim config"
            "\n    -n    number of parallel elements/windows (default=4096 for 16x16x16 tiles)"
            "\n    -m    enable more debug prints (default = false)"
            "\n    -h    show this help message"
            "\n");
}

struct Params getInputParams(int argc, char **argv)
{
    struct Params p;
    p.dramConfigFile = nullptr;
    p.shouldVerify = false;
    p.moreDebugPrints = false;
    p.numElements = 4096; // Default ON-CNN configuration (16 tiles * 16 rows * 16 cols = 4096 PEs)

    int opt;
    while ((opt = getopt(argc, argv, "c:n:m:h")) >= 0)
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
        case 'n':
            p.numElements = atoi(optarg);
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
// =====================================================================
void conv2d_on_cnn(int C_in, int K, int numBits, int numElements, bool debugPrints)
{
    // -----------------------------------------------------------------
    // Step 1: Host Data Preparation (Dummy Data for Benchmark)
    // -----------------------------------------------------------------
    if (debugPrints)
    {
        cout << "[ON-CNN] Preparing Host data vectors..." << endl;
    }

    vector<int16_t> host_ifm(numElements, 2);    // Dummy IFM data
    vector<int16_t> host_filter(numElements, 3); // Dummy Filter data
    vector<int32_t> host_output(numElements, 0); // To store final results

    // -----------------------------------------------------------------
    // Step 2: Memory Allocation in Bit-Serial PIM (Vertical Layout)
    // -----------------------------------------------------------------
    // Allocating objects in the SAME subarray ensures no inter-bank data movement (GDL overhead)
    PimObjId srcX = pimAlloc(PIM_ALLOC_V1, numElements, PIM_INT16);
    PimObjId srcY = pimAllocAssociated(srcX, PIM_INT16);

    // Outputs of custom hardware blocks
    PimObjId destP = pimAllocAssociated(srcX, PIM_INT32);   // OSSM outputs (-1, 0, 1)
    PimObjId destSum = pimAllocAssociated(srcX, PIM_INT32); // OA Tree outputs
    PimObjId q_mag = pimAllocAssociated(srcX, PIM_BOOL);    // Magnitude for OFC
    PimObjId q_sign = pimAllocAssociated(srcX, PIM_BOOL);   // Sign for OFC
    PimObjId destQ = pimAllocAssociated(srcX, PIM_INT32);   // Primary OFC Register
    PimObjId destQM = pimAllocAssociated(srcX, PIM_INT32);  // Secondary OFC Register
    PimObjId acc = pimAllocAssociated(srcX, PIM_INT32);     // Final Accumulator

    // Initialize the Accumulator and OFC registers
    pimBroadcastUInt(destQ, 0);
    pimBroadcastUInt(destQM, 0);
    pimBroadcastUInt(acc, 0);

    // -----------------------------------------------------------------
    // Step 3: Algorithm Loop execution (Mapped to hardware limits)
    // -----------------------------------------------------------------
    // ON-CNN PEs have M=16 multipliers inside them. They process 16 channels at a time.
    int M = 16;
    int total_mac_ops = C_in * K * K;
    int iterations = total_mac_ops / M; // Number of cycles needed for one full PE computation

    if (iterations == 0)
        iterations = 1; // Fallback for very small convolutions

    cout << "........starting ON-CNN MSDF Loop........\n";
    if (debugPrints)
    {
        cout << "[ON-CNN] PE Config: M=16 multipliers | Total MAC Ops per Window = " << total_mac_ops << endl;
        cout << "[ON-CNN] PIM Iterations required = " << iterations << endl;
    }

    auto start = chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter)
    {

        // 3a. Move Data from Host to DRAM Subarrays
        pimCopyHostToDevice((void *)host_ifm.data(), srcX);
        pimCopyHostToDevice((void *)host_filter.data(), srcY);

        // -------------------------------------------------------------
        // HARDWARE BLOCK 1: Online Serial-Serial Multiplier (OSSM)
        // -------------------------------------------------------------
        // Generates the partial products in a signed-digit format bit-by-bit
        pimOSSM(srcX, srcY, destP, numBits);

        // -------------------------------------------------------------
        // HARDWARE BLOCK 2: Online Adder Tree (OA Tree)
        // -------------------------------------------------------------
        // Reduces 16 multiplier outputs within the subarray using Popcounts
        pimOATreeReduce(destP, destSum);

        // -------------------------------------------------------------
        // HARDWARE BLOCK 3: On-the-Fly Converter (OFC)
        // -------------------------------------------------------------
        // In the hardware, this runs concurrently. For benchmarking, we
        // simulate the assimilation over the full generated bit-width.

        // Extract dummy sign and magnitude from the OA tree reduction output
        // Note: 31st bit is sign in 32-bit INT, 0th bit used as dummy magnitude here
        pimBitSliceExtract(destSum, q_sign, 31);
        pimBitSliceExtract(destSum, q_mag, 0);

        // Assimilate the signed-digits into conventional binary (BNS)
        // Online Delay = delta_M (3) + delta_A (2) = 5. So we need numBits + 5 steps.
        int total_steps = numBits + 5;
        for (int step = 0; step < total_steps; ++step)
        {
            pimOFC(q_mag, q_sign, destQ, destQM, step);
        }

        // -------------------------------------------------------------
        // HARDWARE BLOCK 4: Final Accumulator (CPA)
        // -------------------------------------------------------------
        // Accumulate the BNS output of this chunk into the total accumulator
        pimAdd(acc, destQ, acc);
    }

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double, milli> elapsed = end - start;

    cout << "........ending ON-CNN MSDF Loop........\n";
    cout << "[ON-CNN] Processing Time (Host Perspective): " << fixed << setprecision(3) << elapsed.count() << " ms" << endl;

    // Optional: Copy results back to host for verification
    pimCopyDeviceToHost(acc, (void *)host_output.data());

    // --- CPU Verification ---
    cout << "Verifying results with CPU..." << endl;
    bool is_correct = true;

    // شبیه‌سازی منطق کانولوشن روی CPU برای یک پنجره (M = 16)
    int expected_mac = 0;
    for (int iter = 0; iter < iterations; ++iter)
    {
        // در این مثال ساده، تصویر=2 و فیلتر=3 است
        expected_mac += (2 * 3);
    }

    // بررسی خروجی PIM
    for (int i = 0; i < numElements; ++i)
    {
        if (host_output[i] != expected_mac)
        {
            is_correct = false;
            cout << "Mismatch at index " << i << "! Expected: " << expected_mac
                 << ", Got: " << host_output[i] << endl;
            break;
        }
    }

    if (is_correct)
    {
        cout << "SUCCESS: PIM results match CPU exactly!" << endl;
    }

    // -----------------------------------------------------------------
    // Step 4: Cleanup PIM Resources
    // -----------------------------------------------------------------
    pimFree(srcX);
    pimFree(srcY);
    pimFree(destP);
    pimFree(destSum);
    pimFree(q_mag);
    pimFree(q_sign);
    pimFree(destQ);
    pimFree(destQM);
    pimFree(acc);
}

// =====================================================================
// Main Execution
// =====================================================================
int main(int argc, char *argv[])
{

    // Parse user inputs
    struct Params params = getInputParams(argc, argv);

    // Initialize the PIM Device (DRAM-AP Bit-Serial Target)
    if (!createDevice(params.dramConfigFile))
    {
        cerr << "Failed to create PIM device. Check config file." << endl;
        return 1;
    }

    // ON-CNN Network Parameters (Simulating a VGG-16 / ResNet layer)
    // E.g., CONV2_2: Cin = 64, Filter = 3x3
    int C_in = 64;
    int K = 3;
    int numBits = 16; // 16-bit precision based on paper spec

    cout << "========================================================\n";
    cout << "         ON-CNN Hardware Benchmark on PIMeval           \n";
    cout << "========================================================\n";

    // Start PIM Hardware Timer (Internal to PIMeval for profiling)
    pimStartTimer();

    // Execute the main CNN logic mapped to PIM
    conv2d_on_cnn(C_in, K, numBits, params.numElements, params.moreDebugPrints);

    // Stop PIM Hardware Timer
    pimEndTimer();

    // Print Final Hardware Latency & Energy Statistics
    // This will pull data from pimPerfEnergyTables.cpp based on Phase 3!
    pimShowStats();

    cout << "\nBenchmark execution successfully completed." << endl;

    return 0;
}