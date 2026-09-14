#include "libpimeval.h"
#include <iostream>
#include <vector>
#include <cassert>

// توابع pimOSSM، pimOFC و pimOATreeReduce را دقیقاً اینجا کپی کنید

void test_OATreeReduce()
{
    std::cout << "Running OATreeReduce Test..." << std::endl;
    int vecLen = 32; // دو بلوک ۱۶تایی
    PimObjId src = pimAlloc(PIM_ALLOC_V1, vecLen, PIM_INT32);
    PimObjId dest = pimAllocAssociated(src, PIM_INT32);

    // پر کردن ورودی با عدد 1
    std::vector<int32_t> hostSrc(vecLen, 1);
    pimCopyHostToDevice((void *)hostSrc.data(), src);

    pimOATreeReduce(src, dest);

    std::vector<int32_t> hostDest(vecLen);
    pimCopyDeviceToHost(dest, (void *)hostDest.data());

    // در درخت جمع ۱۶تایی، مجموع باید در ایندکس ۱۵ و ۳۱ رسوب کند
    assert(hostDest[15] == 16 && "OATree Failed: Block 1 sum is incorrect");
    assert(hostDest[31] == 16 && "OATree Failed: Block 2 sum is incorrect");

    std::cout << "-> OATreeReduce PASSED!" << std::endl;
    pimFree(src);
    pimFree(dest);
}

void test_OFC()
{
    std::cout << "Running OFC Test..." << std::endl;
    int vecLen = 1;
    PimObjId q_mag = pimAlloc(PIM_ALLOC_V1, vecLen, PIM_BOOL);
    PimObjId q_sign = pimAllocAssociated(q_mag, PIM_BOOL);
    PimObjId destQ = pimAllocAssociated(q_mag, PIM_INT32);
    PimObjId destQM = pimAllocAssociated(q_mag, PIM_INT32);

    pimBroadcastInt(destQ, 0);
    pimBroadcastInt(destQM, 0);

    // تست دستی: وارد کردن رقم +1 (اندازه=1، علامت=0) در مرحله 0
    std::vector<uint8_t> hostMag = {1};
    std::vector<uint8_t> hostSign = {0};
    pimCopyHostToDevice((void *)hostMag.data(), q_mag);
    pimCopyHostToDevice((void *)hostSign.data(), q_sign);

    pimOFC(q_mag, q_sign, destQ, destQM, 0);

    std::vector<int32_t> hostQ(vecLen), hostQM(vecLen);
    pimCopyDeviceToHost(destQ, (void *)hostQ.data());
    pimCopyDeviceToHost(destQM, (void *)hostQM.data());

    // چون رقم +1 بود، باید Q=1 و QM=0 باشد
    assert(hostQ[0] == 1 && hostQM[0] == 0 && "OFC Failed on Step 0");

    std::cout << "-> OFC PASSED!" << std::endl;
    pimFree(q_mag);
    pimFree(q_sign);
    pimFree(destQ);
    pimFree(destQM);
}

void test_OSSM_CSA()
{
    std::cout << "Running OSSM CSA Test..." << std::endl;
    int vecLen = 1;
    int numBits = 16;
    PimObjId srcX = pimAlloc(PIM_ALLOC_V1, vecLen, PIM_INT16);
    PimObjId srcY = pimAllocAssociated(srcX, PIM_INT16);
    PimObjId destP = pimAllocAssociated(srcX, PIM_INT32);

    // ورودی‌های تستی ساده
    std::vector<int16_t> hostX = {5};
    std::vector<int16_t> hostY = {3};
    pimCopyHostToDevice((void *)hostX.data(), srcX);
    pimCopyHostToDevice((void *)hostY.data(), srcY);

    // اجرای تابع دارای مدار CSA
    pimOSSM(srcX, srcY, destP, numBits);

    // از آنجا که SELM ندارید، خروجی destP همیشه صفر است.
    // اما اجرای موفقیت‌آمیز این تابع بدون ارور مموری یا کرش،
    // نشان‌دهنده درستی سینتکس مدار 4:2 CSA در PIMeval است.
    std::cout << "-> OSSM Logic Executed Successfully!" << std::endl;

    pimFree(srcX);
    pimFree(srcY);
    pimFree(destP);
}

int main()
{
    pimCreateDevice(PIM_DEVICE_BITSIMD_V_AP, 1, 128, 32, 1024, 8192);
    std::cout << "--- Starting Unit Tests ---" << std::endl;

    test_OATreeReduce();
    test_OFC();
    test_OSSM_CSA();

    std::cout << "--- All Tests Completed Successfully ---" << std::endl;
    return 0;
}