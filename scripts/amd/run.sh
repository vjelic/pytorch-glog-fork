set -e

# build pytorch
sh scripts/amd/build.sh
sh scripts/amd/test.sh