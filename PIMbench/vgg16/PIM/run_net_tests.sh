#!/usr/bin/env bash
# usage:  BIN=./test_net.out CFG=/path/to/config.cfg QUANT=new bash run_net_tests.sh
#   QUANT = new (percentile requantizer, default) or old (your original max-shift one)
BIN=${BIN:-./test_net.out}
CFG=${CFG:-/home/erfan/Desktop/pim-oncnn/PIMeval-PIMbench/configs/taco/PIMeval_Fulcrum-8192.cfg}
QUANT=${QUANT:-new}
pass=0; fail=0

unset ONCNN_PACK ONCNN_VARIANT ONCNN_BROADCAST ONCNN_BAD_BROADCAST ONCNN_STRICT

run_case () {   # $1=name  $2=ONCNN_TOTAL_PES ("" = default)  $3=profile  $4=seed
  name=$1; pes=$2; prof=$3; seed=$4
  if [ -n "$pes" ]; then export ONCNN_TOTAL_PES=$pes; else unset ONCNN_TOTAL_PES; fi
  python3 gen_net.py "$prof" "$seed" "$QUANT" > gen_log.txt || { echo "gen_net.py failed"; exit 1; }
  out=$($BIN -c "$CFG" 2>&1)
  if echo "$out" | grep -q "SUCCESS"; then
    echo "PASS  $name"; pass=$((pass+1))
  else
    echo "FAIL  $name"; echo "$out" | grep -E "^stage|mismatch|FAILED|Error|Abort" | head -14; fail=$((fail+1))
  fi
}

# 1. Existing 7 net tests (with default ONCNN_BROADCAST=1)
#        name                                              PEs  profile seed
run_case "A: 3 conv layers, no requant, default PEs"       ""   A  1
run_case "A: same, 64 PEs"                                 64   A  2
run_case "A: same, 8 PEs (many filter/window blocks)"      8    A  3
run_case "B: 4 conv layers, requant active, default PEs"   ""   B  1
run_case "B: same, 64 PEs"                                 64   B  2
run_case "B: same, 8 PEs"                                  8    B  3

# Negative control 1: change the LAST expected number; the test must flag exactly 1 mismatch in the last stage
python3 gen_net.py B 4 "$QUANT" > /dev/null
python3 -c "
t=open('net_expected.txt').read().split(); t[-1]=str(int(t[-1])+1)
open('net_expected.txt','w').write(' '.join(t))"
unset ONCNN_TOTAL_PES
out=$($BIN -c "$CFG" 2>&1)
if echo "$out" | grep -q "FAILED: 1 mismatches, first bad stage = 8"; then
  echo "PASS  negative control (detects a 1-number corruption)"; pass=$((pass+1))
else echo "FAIL  negative control"; fail=$((fail+1)); fi

# 2. Verify ONCNN_BROADCAST=0 and ONCNN_BROADCAST=1 produce identical outputs and check call counts
python3 gen_net.py B 1 "$QUANT" > gen_log.txt
out_b0=$(ONCNN_BROADCAST=0 $BIN -c "$CFG" 2>&1)
out_b1=$(ONCNN_BROADCAST=1 $BIN -c "$CFG" 2>&1)
cnt_b0=$(echo "$out_b0" | awk '/oncnn_mac\.int16\.h/ {print $3}')
cnt_b1=$(echo "$out_b1" | awk '/oncnn_mac\.int16\.h/ {print $3}')
if echo "$out_b0" | grep -q "SUCCESS" && echo "$out_b1" | grep -q "SUCCESS" && [ "$cnt_b0" = "$cnt_b1" ]; then
  echo "PASS  ONCNN_BROADCAST=0 and 1 match bit-for-bit (calls=$cnt_b1)"; pass=$((pass+1))
else
  echo "FAIL  ONCNN_BROADCAST=0 vs 1 equivalence"; fail=$((fail+1))
fi

# 3. PACK=pixel call-count test (net_test: Cin=3 in stage 0 and Cin=20 in stage 7 change chunk count)
export ONCNN_TOTAL_PES=4096
exp_flat=$(ONCNN_PACK=flat python3 gen_net.py B 1 "$QUANT" | awk -F'total_calls=' '/ref_summary:/ {split($2,a," "); print a[1]}')
out_flat=$(ONCNN_PACK=flat $BIN -c "$CFG" 2>&1)
got_flat=$(echo "$out_flat" | awk '/oncnn_mac\.int16\.h/ {print $3}')

exp_pix=$(ONCNN_PACK=pixel python3 gen_net.py B 1 "$QUANT" | awk -F'total_calls=' '/ref_summary:/ {split($2,a," "); print a[1]}')
out_pix=$(ONCNN_PACK=pixel $BIN -c "$CFG" 2>&1)
got_pix=$(echo "$out_pix" | awk '/oncnn_mac\.int16\.h/ {print $3}')
unset ONCNN_TOTAL_PES

if echo "$out_flat" | grep -q "SUCCESS" && echo "$out_pix" | grep -q "SUCCESS" && \
   [ "$got_flat" = "$exp_flat" ] && [ "$got_pix" = "$exp_pix" ] && [ "$got_pix" -gt "$got_flat" ]; then
  echo "PASS  ONCNN_PACK=pixel call-count test (flat=$got_flat, pixel=$got_pix)"; pass=$((pass+1))
else
  echo "FAIL  ONCNN_PACK=pixel call-count test (got flat=$got_flat/$exp_flat, pixel=$got_pix/$exp_pix)"; fail=$((fail+1))
fi

# 4. Negative control 2: break broadcast expansion (ONCNN_BAD_BROADCAST=1) -> must fail
python3 gen_net.py A 1 "$QUANT" > /dev/null
out_bad=$(ONCNN_BROADCAST=1 ONCNN_BAD_BROADCAST=1 $BIN -c "$CFG" 2>&1)
if echo "$out_bad" | grep -q "FAILED:"; then
  echo "PASS  negative control (ONCNN_BAD_BROADCAST=1 breaks pe mapping and fails)"; pass=$((pass+1))
else
  echo "FAIL  negative control (ONCNN_BAD_BROADCAST=1 did not fail)"; fail=$((fail+1))
fi

echo "---- $pass passed, $fail failed ----"
[ $fail -eq 0 ]