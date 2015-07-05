#include "stdio.h"  

const int TILE_SIZE = 256;

/***************************************************************************
 *                                                                         *
 *                            Warp Primitives                              *
 *                                                                         *
 ***************************************************************************/

__device__ 
void warpReduce(volatile unsigned int* sdata, unsigned int tid) 
{
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1]; 
}

__device__
unsigned int warpScan(volatile unsigned int* sdata, unsigned int kind, unsigned int tid)
{
    const int lane = tid & 31;
    if (lane >= 1)
        sdata[tid] += sdata[tid - 1];
    if (lane >= 2)
        sdata[tid] += sdata[tid - 2];
    if (lane >= 4)
        sdata[tid] += sdata[tid - 4];
    if (lane >= 8)
        sdata[tid] += sdata[tid - 8];
    if (lane >= 16)
        sdata[tid] += sdata[tid - 16];
    
    if (kind == 1) // 1 - Inclusive scan
        return sdata[tid];
    else           // 0 - Exclusive scan
        return lane > 0 ? sdata[tid - 1] : 0; 
    
}

/***************************************************************************
 *                                                                         *
 *                           Block Scan on Device                          *
 *                                                                         *
 ***************************************************************************/

__device__
unsigned int blockScan(unsigned int* sdata, unsigned int tid)
{
    const unsigned int lane = tid & 31;
    const unsigned int warpID = tid >> 5;
    
    unsigned int val = warpScan(sdata, 0, tid);
    __syncthreads();
    
    if (lane == 31) sdata[warpID] = sdata[tid];
    __syncthreads();
    
    if (warpID == 0) warpScan(sdata, 1, tid);
    __syncthreads();
    
    if (warpID > 0) val += sdata[warpID - 1];
    __syncthreads();
 
    return val;
}

/***************************************************************************
 *                                                                         *
 *                             The Split Operator                          *
 *                                                                         *
 ***************************************************************************/

__device__
unsigned int split(unsigned int* sdata, unsigned int tid, unsigned int pred, unsigned int blockSize)
{
    unsigned int trueBefore = blockScan(sdata, tid);
    
    __shared__ unsigned int falseTotal;
    if (tid == blockSize - 1)
        falseTotal = trueBefore + pred;
    __syncthreads();
    
    if (pred)
        return trueBefore;
    else
        return tid - trueBefore + falseTotal;
}


/***************************************************************************
 *                                                                         *
 *                                Scatter                                  *
 *                                                                         *
 ***************************************************************************/

__global__
void scatter(unsigned int* d_in, unsigned int* d_in_pos, unsigned int* d_out, unsigned int* d_out_pos, 
             unsigned int numElems, unsigned int* d_scanTable, unsigned int mask, unsigned int shift, 
             unsigned int* d_globScan, unsigned int* d_histoAux)
{
    __shared__ unsigned int sScan[16];
    __shared__ unsigned int sHist[16];
    unsigned int gid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int tid = threadIdx.x;
    
    if (tid < 16)
        sHist[tid] = d_histoAux[blockIdx.x * 16 + tid];
    __syncthreads();
    
    if (gid < numElems)
    {
        unsigned int radix = (d_in[gid] & mask) >> shift;
        __syncthreads();
        
        if (tid < 16)
            sScan[tid] = warpScan(sHist, 0, tid);
        __syncthreads();
    
        unsigned int newPos = d_globScan[radix] + d_scanTable[radix * gridDim.x + blockIdx.x] + tid - sScan[radix];
        
        d_out[newPos] = d_in[gid];
        d_out_pos[newPos] = d_in_pos[gid];
    }
}

/***************************************************************************
 *                                                                         *
 *                         Block Level Radix Sort                          *
 *                                                                         *
 ***************************************************************************/
                   
__global__
void localSort(unsigned int* d_in, unsigned int* d_in_pos, unsigned int numElems, unsigned int* d_out, 
               unsigned int* d_out_pos, unsigned int* d_histoTable, unsigned int mask, unsigned int shift, 
               unsigned int digits)
{
    __shared__ unsigned int scan[TILE_SIZE];          // scanned predVector
    __shared__ unsigned int out[TILE_SIZE];           // output data
    __shared__ unsigned int out_pos[TILE_SIZE];       // output positions
    __shared__ unsigned int shist[17];                // histogram
    __shared__ unsigned int count[TILE_SIZE];
    __shared__ unsigned int val[TILE_SIZE];
    
    unsigned int tin;
    unsigned int tin_pos;
    
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid = threadIdx.x;
    
    if (tid < 16)
        shist[tid] = 0;
    __syncthreads();
    
    if (gid < numElems)
    {    
        tin = d_in[gid];
        tin_pos = d_in_pos[gid];
    }
    else
        tin = 0xFFFFFFFFU;
    __syncthreads();
    
    #pragma unroll 4
    for (unsigned int b = 0; b < 4; ++b) // Bit-level sorting
    {
        unsigned int maskedInput = (tin & mask) >> shift;
        unsigned int local_mask = 1 << b; // local mask to mask globally masked input
        unsigned int pred = !((maskedInput & local_mask) >> b); // flags zeroes in the current bit digit
        scan[tid] = pred; // initialize scan vector
        
        unsigned int scatter = split(scan, tid, pred, blockDim.x);
        out[scatter] =  tin; 
        out_pos[scatter] = tin_pos;
        __syncthreads();
        
        tin = out[tid];
        tin_pos = out_pos[tid];
    }
    
    val[tid] = (tin & mask) >> shift;
    count[tid] = 1;
    #pragma unroll 8
	for (int s = 1; s < 256; s <<= 1)
	{
        __syncthreads();
		if ((tid & ((s << 1) - 1)) == 0) 
		{
			if (val[tid] == val[tid + s])
			{
				count[tid] += count[tid + s];
				count[tid + s] = 0;
			}
        }
	}
    __syncthreads();
    if (count[tid])
        atomicAdd(shist + val[tid], count[tid]);
    __syncthreads();
    if(tid < 16)
        d_histoTable[blockIdx.x * 16 + tid] = shist[tid];
    
    d_out[gid] = tin;
    d_out_pos[gid] = tin_pos;
}


/***************************************************************************
 *                                                                         *
 *                            Stone-Kogge Scan                             *
 *                                                                         *
 ***************************************************************************/

__global__
void blockScanGlob(unsigned int* d_in, unsigned int* d_out, unsigned int* d_out_red, const size_t numElems)
{
    __shared__ unsigned int sdata[1024];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * numElems + tid % numElems;
    
    const unsigned int lane = tid & 31;
    const unsigned int warpID = tid >> 5;
    
    if (tid < numElems)
        sdata[tid] = d_in[tid * 16 + blockIdx.x];
    else
        sdata[tid] = 0;
    
    unsigned int val = warpScan(sdata, 0, tid);
    __syncthreads();
    
    if (lane == 31) sdata[warpID] = sdata[tid];
    __syncthreads();
    
    if (warpID == 0) warpScan(sdata, 1, tid);
    __syncthreads();
    
    if (warpID > 0) val += sdata[warpID - 1];
    __syncthreads();
    
   
    if (tid < numElems)
        d_out[gid] = val;
    
    if (tid == numElems - 1)
        d_out_red[blockIdx.x] = val + d_in[(numElems - 1) * 16 + blockIdx.x];     
}

__global__
void warpScanGlob(unsigned int* d_in, unsigned int* d_out, const size_t numElems)
{
    __shared__ unsigned int sdata[32];
    
    if (threadIdx.x < numElems)
        sdata[threadIdx.x] = d_in[threadIdx.x];
    else
        sdata[threadIdx.x] = 0;
    
    sdata[threadIdx.x] = warpScan(sdata, 0, threadIdx.x);
    
    if (threadIdx.x < numElems)
        d_out[threadIdx.x] = sdata[threadIdx.x];
}

__global__
void combineScans(unsigned int* d_in, unsigned int* d_partialRes)
{
    d_in[blockDim.x * blockIdx.x + threadIdx.x] += d_partialRes[blockIdx.x];
}
  

void radix_sort(unsigned int* const d_inputVals,
                unsigned int* const d_inputPos,
                unsigned int* const d_outputVals,
                unsigned int* const d_outputPos,
                const size_t numElems)
{
    const int numThreads = TILE_SIZE;
    const int numBlocks = (numElems - 1) / numThreads + 1;

    unsigned int *vals_src = d_inputVals;
    unsigned int *pos_src  = d_inputPos;

    unsigned int *vals_dst = d_outputVals;
    unsigned int *pos_dst  = d_outputPos;
    
    const int numBits = 4;
    const int numBins = 1 << numBits;
    
    unsigned int* d_scanTable;
    checkCudaErrors(cudaMalloc(&d_scanTable, numBins * numBlocks * sizeof(unsigned int)));
    
    unsigned int* d_histoTable;
    checkCudaErrors(cudaMalloc(&d_histoTable, numBins * numBlocks * sizeof(unsigned int)));
    
    unsigned int* d_reduction;
    checkCudaErrors(cudaMalloc(&d_reduction, numBins * sizeof(unsigned int)));
    
    for (unsigned int i = 0; i < 8 * numBits; i += numBits) 
    {
        unsigned int mask = (numBins - 1) << i;
        
        localSort <<< numBlocks, numThreads >>> (vals_src, pos_src, numElems, vals_dst, pos_dst, d_histoTable, mask, i, numBits);
        
        blockScanGlob <<< numBins, 1024 >>> (d_histoTable, d_scanTable, d_reduction, numBlocks);
        
        warpScanGlob <<< 1, numBins >>> (d_reduction, d_reduction, numBins);
            
        scatter <<< numBlocks, numThreads >>> (vals_dst, pos_dst, vals_src, pos_src, numElems, d_scanTable, mask, i, d_reduction, d_histoTable);  
    }
    
    unsigned int* temp;
    temp = vals_src;
    vals_src = vals_dst;
    vals_dst = temp;
    temp = pos_dst;
    pos_dst = pos_src;
    pos_src = temp;
}