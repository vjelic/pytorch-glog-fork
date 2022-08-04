ROOT_DIR=$(pwd)
DEFAULT_LOG_DIR=$ROOT_DIR/$(git rev-parse --symbolic-full-name --abbrev-ref HEAD)
LOG_DIR="${1:-$DEFAULT_LOG_DIR}"

FILE_LIST=(
    # test_fx
    # test_fx_experimental
    # test_jit
    # test_jit_autocast
    # test_jit_fuser_te
    # test_mkldnn
    # test_mobile_optimizer
    # test_module_init
    # test_quantization
    # test_tensor_creation_ops
    # test_tensorboard
    test_quantization
    test_jit_fuser_te
    distributed/test_distributed_spawn
)

sudo apt install gdb

for FILE_NAME in "${FILE_LIST[@]}"; do
    BASE_FILE_NAME=$(basename $FILE_NAME)
    gdb -ex "set pagination off" \
        -ex "file python" \
        -ex "run /tmp/pytorch/test/${FILE_NAME}.py --verbose" \
        -ex "bt" \
        -ex "set confirm off" \
        -ex "q" \
        2>&1 | tee $LOG_DIR/${BASE_FILE_NAME}_gdb.log
done
