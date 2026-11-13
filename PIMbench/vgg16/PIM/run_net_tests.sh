#!/usr/bin/env bash
# usage:  BIN=./test_net.out CFG=/path/to/PIMeval_Fulcrum_1PE_4096.cfg QUANT=new bash run_net_tests.sh
#
# CFG must be the fixed-4096-PE configuration: num_ranks=1, num_bank_per_rank=128, num_subarray_per_bank=64,
# num_row_per_subarray=65536, num_col_per_subarray=256  (4096 cores, 1 PE per core).
# The script exports ONCNN_PES_PER_CORE=1 to match, and FAILS if the code prints the placement warning.
BIN=${BIN:-./test_net.out}
CFG=${CFG:-/home/erfan/Desktop/oncnn-pim/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum-8192.cfg}
QUANT=${QUANT:-new}
pass=0; fail=0

unset ONCNN_PACK ONCNN_VARIANT ONCNN_BROADCAST ONCNN_BAD_BROADCAST ONCNN_STRICT ONCNN_CALIB_PCT ONCNN_TOTAL_PES
export ONCNN_PES_PER_CORE=1

ok ()  { echo "PASS  $1"; pass=$((pass+1)); }
bad () { echo "FAIL  $1"; fail=$((fail+1)); }

# number of pimONCNNMac calls the simulator recorded (column 3 of the "oncnn_mac..." line of the stats table)
mac_calls () { echo "$1" | awk '$1 ~ /^oncnn_mac/ {print $3; exit}'; }

# $1=name  $2=ONCNN_TOTAL_PES ("" = 4096)  $3=profile  $4=seed  $5=ONCNN_CALIB_PCT ("" = default)
run_case () {
  name=$1; pes=$2; prof=$3; seed=$4; pct=$5
  if [ -n "$pes" ]; then export ONCNN_TOTAL_PES=$pes; else export ONCNN_TOTAL_PES=4096; fi
  if [ -n "$pct" ]; then export ONCNN_CALIB_PCT=$pct; else unset ONCNN_CALIB_PCT; fi
  python3 gen_net.py "$prof" "$seed" "$QUANT" > gen_log.txt || { echo "gen_net.py failed"; cat gen_log.txt; exit 1; }
  out=$($BIN -c "$CFG" 2>&1)
  want=$(cat net_calls.txt); got=$(mac_calls "$out")
  if ! echo "$out" | grep -q "SUCCESS"; then
    bad "$name  (numbers)"; echo "$out" | grep -E "^stage|mismatch|FAILED|Error|Abort" | head -14
  elif echo "$out" | grep -q "ON-CNN WARNING"; then
    bad "$name  (placement warning: wrong CFG for ONCNN_PES_PER_CORE=1)"
  elif [ "$got" != "$want" ]; then
    bad "$name  (pimONCNNMac calls: expected $want, simulator counted '$got')"
  else
    ok "$name  [$got MAC calls]"
  fi
  unset ONCNN_CALIB_PCT
}

#        name                                                     PEs   profile seed  calib_pct
run_case "A: 3 conv layers, no requant, 4096 PEs"                 ""    A  1
run_case "A: same, 64 PEs"                                        64    A  2
run_case "A: same, 8 PEs (many filter/window blocks)"             8     A  3
run_case "B: 4 conv layers, requant active, 4096 PEs"             ""    B  1
run_case "B: same, 64 PEs"                                        64    B  2
run_case "B: same, 8 PEs"                                         8     B  3
run_case "C: 300 filters, 400 windows, Cin=300, 2x3 kernel, 4096 PEs (2 filter blocks)" "" C 1
run_case "C: same, 64 PEs"                                        64    C  2
run_case "C: same, 100 PEs (grid not a power of two)"             100   C  3
run_case "D: requant with saturation (calib 0.99), 4096 PEs"      ""    D  1  0.99
run_case "D: same, 64 PEs"                                        64    D  2  0.99

# ---- negative controls: the checks above must be able to FAIL -------------------------------------
export ONCNN_TOTAL_PES=4096; unset ONCNN_CALIB_PCT
python3 gen_net.py B 4 "$QUANT" > /dev/null
nstages=$(( $(wc -l < net_meta.txt) - 1 ))          # meta file = 1 input line + one line per stage

# 1. one wrong expected number in the last stage -> exactly 1 mismatch, reported in the last stage
python3 -c "
t=open('net_expected.txt').read().split(); t[-1]=str(int(t[-1])+1)
open('net_expected.txt','w').write(' '.join(t))"
out=$($BIN -c "$CFG" 2>&1)
if echo "$out" | grep -q "FAILED: 1 mismatches, first bad stage = $((nstages-1))"; then ok "negative control 1 (1-number corruption of the expected output)"
else bad "negative control 1"; fi

# 2. one wrong WEIGHT (expected output untouched) -> the convolution result must change and be detected
python3 gen_net.py B 4 "$QUANT" > /dev/null
python3 -c "
t=open('net_weights.txt').read().split(); t[0]=str(int(t[0])+3)
open('net_weights.txt','w').write(' '.join(t))"
out=$($BIN -c "$CFG" 2>&1)
if echo "$out" | grep -q "FAILED" && ! echo "$out" | grep -q "SUCCESS"; then ok "negative control 2 (corrupted weight is detected)"
else bad "negative control 2 (a wrong weight was NOT detected - the test does not exercise the weights)"; fi

# 3. the call-count check itself must be able to fail
python3 gen_net.py A 1 "$QUANT" > /dev/null
out=$($BIN -c "$CFG" 2>&1); want=$(( $(cat net_calls.txt) + 1 )); got=$(mac_calls "$out")
if [ -n "$got" ] && [ "$got" != "$want" ]; then ok "negative control 3 (call-count check detects a wrong expected count)"
else bad "negative control 3"; fi

# 4. the placement warning must really fire when PEs-per-core does not match the config
out=$(ONCNN_PES_PER_CORE=4 $BIN -c "$CFG" 2>&1)
if echo "$out" | grep -q "ON-CNN WARNING"; then ok "negative control 4 (placement warning fires on a mismatched ONCNN_PES_PER_CORE)"
else bad "negative control 4"; fi

echo "---- $pass passed, $fail failed ----"
[ $fail -eq 0 ]