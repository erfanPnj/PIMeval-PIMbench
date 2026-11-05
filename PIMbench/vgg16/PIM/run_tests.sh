#!/usr/bin/env bash
# usage:  BIN=./test_conv.out CFG=/path/to/config.cfg bash run_tests.sh
BIN=${BIN:-./test_conv.out}
CFG=${CFG:-/home/erfan/Desktop/pim-oncnn/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum_Bank128-8192.cfg}
pass=0; fail=0

unset ONCNN_PACK ONCNN_VARIANT ONCNN_BROADCAST ONCNN_BAD_BROADCAST ONCNN_STRICT ONCNN_PRINT_EQ10

run_case () {   # $1=name  $2=ONCNN_TOTAL_PES ("" = default)  $3..=gen.py args
  name=$1; pes=$2; shift 2
  python3 gen.py "$@" || { echo "gen.py failed"; exit 1; }
  if [ -n "$pes" ]; then export ONCNN_TOTAL_PES=$pes; else unset ONCNN_TOTAL_PES; fi
  out=$($BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
  if echo "$out" | grep -q "SUCCESS"; then
    echo "PASS  $name"; pass=$((pass+1))
  else
    echo "FAIL  $name"; echo "$out" | grep -E "Mismatch|FAILED|Error|Abort" | head -5; fail=$((fail+1))
  fi
}

#        name                                   PEs   cin cout h w kh kw s p  lo  hi   seed
run_case "your original case"                   ""    3  4  8 8 3 3 1 1  -5   5   1
run_case "non-square input, partial window blk" ""    3  5  7 9 3 3 1 1  -5   5   2
run_case "several filter blocks (8 PEs)"        8     3  5  7 9 3 3 1 1  -5   5   3
run_case "3x5 kernel, stride 2, pad 2"          16    5  4  9 7 3 5 2 2  -5   5   4
run_case "Cin=20, chunks cut mid-pixel"         12   20  6  6 6 3 3 1 1  -5   5   5
run_case "large values +-1000 (acc width)"      ""   16  3  5 5 3 3 1 1 -1000 1000 6

# Negative control: corrupt ONE number, the test MUST report exactly 1 mismatch
python3 gen.py 3 4 8 8 3 3 1 1 -5 5 7
python3 -c "
t=open('test_output_expected.txt').read().split(); t[0]=str(int(t[0])+1)
open('test_output_expected.txt','w').write(' '.join(t))"
unset ONCNN_TOTAL_PES
out=$($BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
if echo "$out" | grep -q "Found 1 mismatches"; then echo "PASS  negative control (detects a 1-number corruption)"; pass=$((pass+1))
else echo "FAIL  negative control"; fail=$((fail+1)); fi

# Broadcast H2D byte check + ONCNN_BROADCAST=0 vs 1 equivalence on Cin=3, Cout=4, 8x8, 3x3
# curF=4, curW=16 across 4 window blocks (64 windows), numChunks=2 (flat):
#   Zero-init Acc H2D = 4 * (4*16*8) = 2048 B
#   Unique Operands H2D = 4 window blocks * 2 chunks * (16 + 4) * 16 * 2 = 5120 B
#   Total H2D = 7168 B
python3 gen.py 3 4 8 8 3 3 1 1 -5 5 1
out_b1=$(ONCNN_TOTAL_PES=4096 ONCNN_BROADCAST=1 ONCNN_PACK=flat $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
out_b0=$(ONCNN_TOTAL_PES=4096 ONCNN_BROADCAST=0 ONCNN_PACK=flat $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
h2d_b1=$(echo "$out_b1" | awk '/Host to Device/ {print $5}')
if echo "$out_b1" | grep -q "SUCCESS" && echo "$out_b0" | grep -q "SUCCESS" && [ "$h2d_b1" = "7168" ]; then
  echo "PASS  broadcast H2D unique bytes exact (7168 B = 2048 acc + 5120 unique operands)"; pass=$((pass+1))
else
  echo "FAIL  broadcast H2D unique bytes (got $h2d_b1, expected 7168)"; fail=$((fail+1))
fi

# PACK=pixel call-count & Eq.10 test on Cin=3, Cout=4, 8x8, 3x3 (4 window blocks: flat=8 calls, pixel=36 calls)
out_pix=$(ONCNN_TOTAL_PES=4096 ONCNN_PACK=pixel $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
cnt_flat=$(echo "$out_b1" | awk '/oncnn_mac\.int16\.h/ {print $3}')
cnt_pix=$(echo "$out_pix" | awk '/oncnn_mac\.int16\.h/ {print $3}')
if echo "$out_pix" | grep -q "SUCCESS" && [ "$cnt_flat" = "8" ] && [ "$cnt_pix" = "36" ]; then
  echo "PASS  ONCNN_PACK=pixel single-layer call count (flat=8, pixel=36)"; pass=$((pass+1))
else
  echo "FAIL  ONCNN_PACK=pixel single-layer call count (flat=$cnt_flat, pixel=$cnt_pix)"; fail=$((fail+1))
fi

# ONCNN_PRINT_EQ10 verification on ResNet-18 CONV2-2 (Cout=64, 56x56, Cin=64, 3x3): 338,688 (SS) and 331,632 (SP)
python3 gen.py 64 64 56 56 3 3 1 1 -1 1 8
out_ss=$(ONCNN_TOTAL_PES=4096 ONCNN_VARIANT=SS ONCNN_PACK=flat ONCNN_PRINT_EQ10=1 $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
out_sp=$(ONCNN_TOTAL_PES=4096 ONCNN_VARIANT=SP ONCNN_PACK=flat ONCNN_PRINT_EQ10=1 $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
if echo "$out_ss" | grep -q "338688 cycles" && echo "$out_sp" | grep -q "331632 cycles"; then
  echo "PASS  ONCNN_PRINT_EQ10 reproduces ResNet-18 CONV2-2 cycles (SS=338688, SP=331632)"; pass=$((pass+1))
else
  echo "FAIL  ONCNN_PRINT_EQ10 CONV2-2 verification"; fail=$((fail+1))
fi

# Negative control: broken broadcast expansion (ONCNN_BAD_BROADCAST=1) must fail
python3 gen.py 3 4 8 8 3 3 1 1 -5 5 1
out_bad=$(ONCNN_BROADCAST=1 ONCNN_BAD_BROADCAST=1 $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
if echo "$out_bad" | grep -q "FAILED:"; then
  echo "PASS  negative control (ONCNN_BAD_BROADCAST=1 breaks broadcast pe mapping and fails)"; pass=$((pass+1))
else
  echo "FAIL  negative control (ONCNN_BAD_BROADCAST=1 did not fail)"; fail=$((fail+1))
fi

# int32 accumulator overflow detection test:
# Cin=1, Cout=1, H=3, W=3, KH=3, KW=3, s=1, p=0 with int16-valid inputs=20000 and weights=20000.
# 9 * 20000 * 20000 = 3,600,000,000 > INT32_MAX (2,147,483,647): fits in int16 operands & int64 acc, overflows int32.
echo "1 1 3 3 3 3 1 0" > test_meta.txt
echo "20000 20000 20000 20000 20000 20000 20000 20000 20000" > test_input.txt
echo "20000 20000 20000 20000 20000 20000 20000 20000 20000" > test_weight.txt
echo "0" > test_output_expected.txt
out_ovf0=$(ONCNN_STRICT=0 $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
out_ovf1=$(ONCNN_STRICT=1 $BIN -c "$CFG" -i test_input.txt -k test_weight.txt -e test_output_expected.txt 2>&1)
rc_ovf1=$?
if echo "$out_ovf0" | grep -q "int32 accumulator overflow count=1" && \
   echo "$out_ovf1" | grep -q "ON-CNN Abort: accumulator overflowed int32" && [ $rc_ovf1 -ne 0 ]; then
  echo "PASS  int32 accumulator overflow detection (count reported & ONCNN_STRICT=1 aborts)"; pass=$((pass+1))
else
  echo "FAIL  int32 accumulator overflow detection"; fail=$((fail+1))
fi

echo "---- $pass passed, $fail failed ----"
[ $fail -eq 0 ]