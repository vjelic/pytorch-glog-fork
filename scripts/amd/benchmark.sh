cd benchmarks/operator_benchmark

# install op benchmark
cd pt_extension
python setup.py install
cd ..

python -m pt.sum_test