#pragma once
#include"cuda_runtime.h"
//Morton编码 调整无序点云
struct Morton
{
	const static int levels = 10;
	const static int bits_per_level = 3;
	const static int nbits = levels * bits_per_level;

	using code_t = int;

	__device__ __host__ __forceinline__
		static int spreadBits(int x, int offset)
	{
		//......................9876543210
		x = (x | (x << 10)) & 0x000f801f; //............98765..........43210
		x = (x | (x << 4)) & 0x00e181c3; //........987....56......432....10
		x = (x | (x << 2)) & 0x03248649; //......98..7..5..6....43..2..1..0
		x = (x | (x << 2)) & 0x09249249; //....9..8..7..5..6..4..3..2..1..0

		return x << offset;
	}

	__device__ __host__ __forceinline__
		static int compactBits(int x, int offset)
	{
		x = (x >> offset) & 0x09249249;  //....9..8..7..5..6..4..3..2..1..0
		x = (x | (x >> 2)) & 0x03248649;  //......98..7..5..6....43..2..1..0                                          
		x = (x | (x >> 2)) & 0x00e181c3;  //........987....56......432....10                                       
		x = (x | (x >> 4)) & 0x000f801f;  //............98765..........43210                                          
		x = (x | (x >> 10)) & 0x000003FF;  //......................9876543210        

		return x;
	}

	__device__ __host__ __forceinline__
		static code_t createCode(int cell_x, int cell_y, int cell_z)
	{
		return spreadBits(cell_x, 0) | spreadBits(cell_y, 1) | spreadBits(cell_z, 2);
	}

	__device__ __host__ __forceinline__
		static void decomposeCode(code_t code, int& cell_x, int& cell_y, int& cell_z)
	{
		cell_x = compactBits(code, 0);
		cell_y = compactBits(code, 1);
		cell_z = compactBits(code, 2);
	}

	__device__ __host__ __forceinline__
		static uint3 decomposeCode(code_t code)
	{
		return make_uint3(compactBits(code, 0), compactBits(code, 1), compactBits(code, 2));
	}
	//每3个bit为一层，一层刚好为0~7，共八个节点
	__host__ __device__ __forceinline__
		static code_t extractLevelCode(code_t code, int level)
	{
		return (code >> (nbits - 3 * (level + 1))) & 7;
	}

	__host__ __device__ __forceinline__
		static code_t shiftLevelCode(code_t level_code, int level)
	{
		return level_code << (nbits - 3 * (level + 1));
	}
};

struct CalcMorton
{
	const static int depth_mult = 1 << Morton::levels;

	float3 minp_;
	float3 dims_;

	__device__ __host__ __forceinline__ CalcMorton(float3 minp, float3 maxp) : minp_(minp)
	{
		dims_.x = maxp.x - minp.x;
		dims_.y = maxp.y - minp.y;
		dims_.z = maxp.z - minp.z;
	}

	__device__ __host__ __forceinline__ Morton::code_t operator()(const float3& p) const
	{
		const int cellx = min((int)floorf(depth_mult * min(1.f, max(0.f, (p.x - minp_.x) / dims_.x))), depth_mult - 1);
		const int celly = min((int)floorf(depth_mult * min(1.f, max(0.f, (p.y - minp_.y) / dims_.y))), depth_mult - 1);
		const int cellz = min((int)floorf(depth_mult * min(1.f, max(0.f, (p.z - minp_.z) / dims_.z))), depth_mult - 1);

		return Morton::createCode(cellx, celly, cellz);
	}
	__device__ __host__ __forceinline__ Morton::code_t operator()(const float4& p) const
	{
		return (*this)(make_float3(p.x, p.y, p.z));
	}
};
template<typename Iterator, typename T, typename BinaryPredicate>
__host__ __device__ Iterator Lower_bound(Iterator first, Iterator last, const T& val, BinaryPredicate comp)
{
	int len = last - first;

	while (len > 0)
	{
		int half = len >> 1;
		Iterator middle = first;

		middle += half;

		if (comp(*middle, val))
		{
			first = middle;
			++first;
			len -= half + 1;
		}
		else
		{
			len = half;
		}
	}
	return first;
}
struct CompareByLevelCode
{
	int level;

	__device__ __host__ __forceinline__
		CompareByLevelCode(int level_arg) : level(level_arg) {}

	__device__ __host__ __forceinline__
		bool operator()(Morton::code_t code1, Morton::code_t code2) const
	{
		return Morton::extractLevelCode(code1, level) < Morton::extractLevelCode(code2, level);
	}
};
__device__ __host__ __forceinline__
static bool checkIfNodeInsideSphere(const float3& minp, const float3& maxp, const float3& c, float r)//检查结点是否在半球内部
{
	r *= r;

	float d2_xmin = (minp.x - c.x) * (minp.x - c.x);
	float d2_ymin = (minp.y - c.y) * (minp.y - c.y);
	float d2_zmin = (minp.z - c.z) * (minp.z - c.z);

	if (d2_xmin + d2_ymin + d2_zmin > r)
		return false;

	float d2_zmax = (maxp.z - c.z) * (maxp.z - c.z);

	if (d2_xmin + d2_ymin + d2_zmax > r)
		return false;

	float d2_ymax = (maxp.y - c.y) * (maxp.y - c.y);

	if (d2_xmin + d2_ymax + d2_zmin > r)
		return false;

	if (d2_xmin + d2_ymax + d2_zmax > r)
		return false;

	float d2_xmax = (maxp.x - c.x) * (maxp.x - c.x);

	if (d2_xmax + d2_ymin + d2_zmin > r)
		return false;

	if (d2_xmax + d2_ymin + d2_zmax > r)
		return false;

	if (d2_xmax + d2_ymax + d2_zmin > r)
		return false;

	if (d2_xmax + d2_ymax + d2_zmax > r)
		return false;

	return true;
}

__device__ __host__ __forceinline__
static bool checkIfNodeOutsideSphere(const float3& minp, const float3& maxp, const float3& c, float r)
{
	if (maxp.x < (c.x - r) || maxp.y < (c.y - r) || maxp.z < (c.z - r))
		return true;

	if ((c.x + r) < minp.x || (c.y + r) < minp.y || (c.z + r) < minp.z)
		return true;

	return false;
}

__device__ __host__ __forceinline__
static void calcBoundingBox(int level, int code, float3& res_minp, float3& res_maxp)
{
	int cell_x, cell_y, cell_z;
	Morton::decomposeCode(code, cell_x, cell_y, cell_z);

	float cell_size_x = (res_maxp.x - res_minp.x) / (1 << level);
	float cell_size_y = (res_maxp.y - res_minp.y) / (1 << level);
	float cell_size_z = (res_maxp.z - res_minp.z) / (1 << level);

	res_minp.x += cell_x * cell_size_x;
	res_minp.y += cell_y * cell_size_y;
	res_minp.z += cell_z * cell_size_z;

	res_maxp.x = res_minp.x + cell_size_x;
	res_maxp.y = res_minp.y + cell_size_y;
	res_maxp.z = res_minp.z + cell_size_z;
}