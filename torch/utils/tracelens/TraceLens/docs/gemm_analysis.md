## GEMM Efficiency Analysis


Whats a gemm?
General matrix multiply 
 
C=α⋅A@B+β⋅C
 
•	We define a gemm problem size primarly as M N K 
Where M = A.shape[0]
K the inner product dimention = A.shape[1] = B.shape[0]
N = B.shape[1]
We have efficient implementations of GEMM n the kernel libraries (*blas) so we should always translate a computation into matmul and call the kernel libraries.


 
Now say we have an LLM (very common so taking as an example)
The LLM has lot of linear layers in MLP we have up proj from d_model to d_ff and down proj from d_ff to 
Let's take the example of the up proj Linear layer which inputs tensors shape B, L d_model where b and l are batch size and sequence length
And outputs tensor shape B, L, d_ff
 for each request in batch size and token in seq length we do X[b, l, :] @ W^T
Where W is the weight matrix from the linear layer with shape d_ff, d_model
This sequence of computation can be written as a single matmul by flattening the matrix X across the batch and sequence axes. You can also thinking of it as stacking the B*L feature row vectors to get a matrix of shape B*L, d_model
 
Now the op is 
Y = X_flatten @ W^T
 
Lets write the M N K tuple for this 
M = X_flatten.shape[0] = B*L
K = X_flatten.shape[0] = d_model
N = W^T.shape[1] = W.shape[0] = d_ff
So our M N K tuple is (B*L, d_model, d_ff)
 
Prefill vs decode:
In prefill we pass the full sequence through the model 
While in decode we cache the L-1 and compute only for the current token at index L
The linear layer therefore sees input tensor of shape B,1,d_model
As a result our GEMM M N K tuple is (B, d_model, d_ff) without dependency on sequence length
So how much ever long the seq is the GEMM problem sze is the same. The dreaded O(L) scaling comes from the attention op which we cover in future. 
 
To summarize in prefill we see 
 	Prefill	Decode
M	B*L	B
N	d_model	d_model
K	d_ff	d_ff

in general we flatten all but the last axis of the input tensor to get the M dimension
The last axis is the K dimension and the output tensor is always the N dimension


backward of Y = X @ W^T
We have
X_grad = Y_grad @ (W)
W_grad = Y_grad^T @ (X)
B_grad = Y_grad.sum(dim=0)

So M N K in fwd leads to two gemms in the backward pass with shapes (M, K, N) and (N, K, M)


### Performance analysis

How many operations in matrix multiply
C[i,j] = sum over k { A[I,k] * B[k,j] }
For each element of output we have k multiplies and k-1 additions
So total 2k-1 operations per output element
In total we have M*N output elements
So we have M*N*(2K-1)
In practice 2k >> 1
So we can just write num ops = 2*M*N*K
 
What about memory read writes 
We read in A and B of size M,K and K*N and write C of size M*N
So total memory traffic is (M*K + K*N + M*N)* bytes_per_elem
 
 
Now lets compute the ops/byte = 2*M*N*K/ (M*K + K*N + M*K)*bytes_per_elem

this is the roofline metric we are looking for

Primer on roofline:
We have compute engine and memory to push data in and out. Note we don’t assume any detail about the comoute engine here . It can be gpu cpu, whateve 
 
 
Similarly if rate at which data can be pushed in and out is more thent he compute rate then we are  limited by the compute rate
"perf" is usually the compute rate , flops/s or more generally ops/s
 
In this case the "perf" we see would be bounded by the compute rate. Now how much bug is problem size after this we get same performance ie in this regime the perf roof is not dpendent on the problem size
 
If the rate at which data can be pushed in and out is lesser than the rate at which the compute engine can process the data then the memory is the rate limiter  and we call the memory bound process
This 
 
In the memory bound case let's think about the upper bound of "perf"
rate at which data can be transferred in/out would be 
Ops/s = ops/byte * bytes/s
 
Ops/byte is dependent on the op type . This is an important metric called the operational intensity this is why we calculated this prec
 
So the perf roof is 
Ops/byte max =  ops/byte * bytes/s max
=  ops/byte * memory bw
 
We saw ops/byte 
Now we saw the "roof" of perf is 
Perf aka ops/s =  compute engine max ops/s as peak perf if compute bound 
 = ops/byte * memory bw if memory bound
 
Now when the transition happens?
When bytes
Bytes/s * ops/byte > ops/s
Mem bw *ops/byte > peak perf
Ops/byte > peak perf / memory bandwidth
 
Compute for M300X this boundary is 1307 TFLOPs/s and 5.3TB/s
 1307/5.3 = 246.6038 flops/byte

So we have following plot of perf roof vs flops / byte
 
caveat
Note that although each element of A is used for computing every element of the row of C 
here we assume the input matrices are read once and we have complete reuse. This is not true in practise but only a first approximation acting as lower bound of memory traffic. The true memory traffic is highly dependent on the implementation detail while this naïve memory traffic is independent of the implementation
In contrast the num of ops is truly independent of implementation 
As discussed this is the lower bound of memory traffic, this is upper bound of op intensity

