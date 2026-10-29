#!/usr/bin/env bash
# usage:  BIN=./test_net.out CFG=/path/to/config.cfg QUANT=new bash run_net_tests.sh
#   QUANT = new (percentile requantizer, default) or old (your original max-shift one)
BIN=${BIN:-./test_net.out}
CFG=${CFG:-/home/erfan/Desktop/pim-oncnn/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum_Bank128-8192.cfg}
QUANT=${QUANT:-new}
pass=0; fail=0

run_case () {   # $1=name  $2=ONCNN_TOTAL_PES ("" = default)  $3=profile  $4=seed
  name=$1; pes=$2; prof=$3; seed=$4
  python3 gen_net.py "$prof" "$seed" "$QUANT" > gen_log.txt || { echo "gen_net.py failed"; exit 1; }
  if [ -n "$pes" ]; then export ONCNN_TOTAL_PES=$pes; else unset ONCNN_TOTAL_PES; fi
  out=$($BIN -c "$CFG" 2>&1)
  if echo "$out" | grep -q "SUCCESS"; then
    echo "PASS  $name"; pass=$((pass+1))
  else
    echo "FAIL  $name"; echo "$out" | grep -E "^stage|mismatch|FAILED|Error|Abort" | head -14; fail=$((fail+1))
  fi
}

#        name                                              PEs  profile seed
run_case "A: 3 conv layers, no requant, default PEs"       ""   A  1
run_case "A: same, 64 PEs"                                 64   A  2
run_case "A: same, 8 PEs (many filter/window blocks)"      8    A  3
run_case "B: 4 conv layers, requant active, default PEs"   ""   B  1
run_case "B: same, 64 PEs"                                 64   B  2
run_case "B: same, 8 PEs"                                  8    B  3

# Negative control: change the LAST expected number; the test must flag exactly 1 mismatch in the last stage
python3 gen_net.py B 4 "$QUANT" > /dev/null
python3 -c "
t=open('net_expected.txt').read().split(); t[-1]=str(int(t[-1])+1)
open('net_expected.txt','w').write(' '.join(t))"
unset ONCNN_TOTAL_PES
out=$($BIN -c "$CFG" 2>&1)
if echo "$out" | grep -q "FAILED: 1 mismatches, first bad stage = 8"; then
  echo "PASS  negative control (detects a 1-number corruption)"; pass=$((pass+1))
else echo "FAIL  negative control"; fail=$((fail+1)); fi

echo "---- $pass passed, $fail failed ----"
[ $fail -eq 0 ]