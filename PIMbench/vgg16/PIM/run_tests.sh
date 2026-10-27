#!/usr/bin/env bash
# usage:  BIN=./test_conv.out CFG=/path/to/config.cfg bash run_tests.sh
BIN=${BIN:-./test_conv.out}
CFG=${CFG:-/home/erfan/Desktop/pim-oncnn/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum_Bank128-8192.cfg}
pass=0; fail=0

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

echo "---- $pass passed, $fail failed ----"