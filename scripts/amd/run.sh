set -e

clear

# build pytorch
bash scripts/amd/prep.sh
bash scripts/amd/build.sh
bash scripts/amd/test.sh