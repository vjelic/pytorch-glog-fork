cd scripts/amd/
OUT_FILE="a.out"
rm -rf ./$OUT_FILE
/opt/rocm/bin/hipcc check_warp.cu -o $OUT_FILE
./$OUT_FILE
