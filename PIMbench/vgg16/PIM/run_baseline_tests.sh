#!/usr/bin/env bash
# usage:  BIN=./test_net.out CFG=/path/to/baseline.cfg bash run_baseline_tests.sh
# Baseline numerical test: compares every stage (conv / relu / pool) with the Python reference.
BIN=${BIN:-./test_net.out}
CFG=${CFG:-/home/erfan/Desktop/oncnn-pim/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum-8192.cfg}
pass=0; fail=0

unset ONCNN_PACK ONCNN_VARIANT ONCNN_BROADCAST ONCNN_STRICT ONCNN_CALIB_PCT ONCNN_TOTAL_PES ONCNN_PES_PER_CORE
export BASELINE=1

ok ()  { echo "PASS  $1"; pass=$((pass+1)); }
bad () { echo "FAIL  $1"; fail=$((fail+1)); }

# $1=name  $2=profile  $3=seed
run_case () {
  python3 gen_net.py "$2" "$3" new > gen_log.txt || { echo "gen_net.py failed"; cat gen_log.txt; exit 1; }
  out=$($BIN -c "$CFG" 2>&1)
  if echo "$out" | grep -q "SUCCESS"; then ok "$1"
  else bad "$1"; echo "$out" | grep -E "^stage|mismatch|FAILED|Error|Abort" | head -14; fi
}

run_case "A: 4D+3D conv, relu, pool, seed 1"                    A 1
run_case "A: seed 2"                                            A 2
run_case "C: 300 filters, Cin=300, stride 2, 2x3 kernel, seed 1" C 1
run_case "C: seed 2"                                            C 2
run_case "D: large values (int32 baseline only), seed 1"        D 1

# ---- negative controls: the checks above must be able to FAIL ----
python3 gen_net.py A 4 new > /dev/null
nstages=$(( $(wc -l < net_meta.txt) - 1 ))

# 1. one wrong expected number in the last stage -> exactly 1 mismatch
python3 -c "
t=open('net_expected.txt').read().split(); t[-1]=str(int(t[-1])+1)
open('net_expected.txt','w').write(' '.join(t))"
out=$($BIN -c "$CFG" 2>&1)
if echo "$out" | grep -q "FAILED: 1 mismatches, first bad stage = $((nstages-1))"; then ok "negative control 1 (corrupted expected output)"
else bad "negative control 1"; fi

# 2. one wrong WEIGHT -> the convolution result must change and be detected
python3 gen_net.py A 4 new > /dev/null
python3 -c "
t=open('net_weights.txt').read().split(); t[0]=str(int(t[0])+3)
open('net_weights.txt','w').write(' '.join(t))"
out=$($BIN -c "$CFG" 2>&1)
if echo "$out" | grep -q "FAILED" && ! echo "$out" | grep -q "SUCCESS"; then ok "negative control 2 (corrupted weight is detected)"
else bad "negative control 2 (a wrong weight was NOT detected)"; fi

echo "---- $pass passed, $fail failed ----"
[ $fail -eq 0 ]