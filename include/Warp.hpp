#pragma once
#include"cuda_runtime.h"
//Ïß³ÌÊøº¯Êý
struct Warp {
	template<typename T>
	static __device__ __forceinline__ T reduce(volatile T* buffer, T init)
	{
		unsigned int idx = Warp::laneId();
		buffer[idx] = init;
		int stride = 32 / 2;
		for (; stride > 0; stride >>= 1)
		{
			if (idx < stride)
			{
				buffer[idx] += buffer[idx + stride];
			}
			__syncthreads();
		}
		return buffer[0];
	}
	static __device__ __forceinline__ unsigned int
		laneId()
	{
		unsigned int ret;
		asm("mov.u32 %0, %laneid;" : "=r"(ret));
		return ret;
	}
	template<typename InIt, typename OutIt>
	static __device__ __forceinline__ OutIt copy(InIt beg, InIt end, OutIt out)
	{
		unsigned int lane = laneId();
		InIt  t = beg + lane;
		OutIt o = out + lane;

		for (; t < end; t += STRIDE, o += STRIDE)
			*o = *t;
		return o;
	}
};

enum ScanKind { exclusive, inclusive };
template <ScanKind Kind, class T>
__device__ __forceinline__ T scan_warp(volatile T* ptr, const unsigned int idx = threadIdx.x)
{
	const unsigned int lane = idx & 31; // index of thread in warp (0..31)

	if (lane >= 1) ptr[idx] = ptr[idx - 1] + ptr[idx];
	if (lane >= 2) ptr[idx] = ptr[idx - 2] + ptr[idx];
	if (lane >= 4) ptr[idx] = ptr[idx - 4] + ptr[idx];
	if (lane >= 8) ptr[idx] = ptr[idx - 8] + ptr[idx];
	if (lane >= 16) ptr[idx] = ptr[idx - 16] + ptr[idx];

	if (Kind == inclusive)
		return ptr[idx];
	else
		return (lane > 0) ? ptr[idx - 1] : 0;
}

template <ScanKind Kind, class T>
__device__ __forceinline__ T scan_block(volatile T* ptr, const unsigned int idx = threadIdx.x)
{
	const unsigned int lane = idx & 31;
	const unsigned int warpid = idx >> 5;

	T val = scan_warp <Kind>(ptr, idx);

	__syncthreads();

	if (lane == 31)
		ptr[warpid] = ptr[idx];

	__syncthreads();

	if (warpid == 0)
		scan_warp<inclusive>(ptr, idx);

	__syncthreads();

	if (warpid > 0)
		val = ptr[warpid - 1] + val;

	__syncthreads();

	ptr[idx] = val;

	__syncthreads();

	return val;
}
