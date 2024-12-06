#pragma once
#include <cstdio>
#include <limits>
#include "math_functions.h"
#include <device_launch_parameters.h>
#include"cuda_runtime.h"
using namespace std;
#include<iostream>
#include"Octree.h"
//点类型运算重载方法

int divUp(int total, int grain) { return (total + grain - 1) / grain; };

template <class T>
__device__ __host__ __forceinline__ void dswap(T& a, T& b)
{
	T c(a); a = b; b = c;
}

__device__ __host__ __forceinline__ float
length(const float3& v)
{
	return v.x*v.x+v.y*v.y+v.z*v.z;
}

__device__ __host__ __forceinline__ float3
cross(const float3& v1, const float3& v2)
{
	return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__device__ __forceinline__ float
dot(const float3& v1, const float3& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}


__device__ __forceinline__ float3
operator+(const float3& v1, const float3& v2)
{
	return make_float3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

__device__ __forceinline__ double3
operator+(const double3& v1, const double3& v2)
{
	return make_double3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}


__device__ __forceinline__ float3&
operator*=(float3& vec, const float& v)
{
	vec.x *= v;  vec.y *= v;  vec.z *= v; return vec;
}

__device__ __forceinline__ float3
operator-(const float3& v1, const float3& v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ __forceinline__ float3
operator-(const double3& v1, const float3& v2)
{
	return make_float3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

__device__ __forceinline__ float3
operator*(const float3& v1, const float& v)
{
	return make_float3(v1.x * v, v1.y * v, v1.z * v);
}


__device__ __forceinline__ float3
operator/(const double3& v1, const int& v)
{
	return make_float3(v1.x / v, v1.y / v, v1.z / v);
}

__device__ __forceinline__ float3
operator/(const float3& v1, const int& v)
{
	return make_float3(v1.x / v, v1.y / v, v1.z / v);
}

__device__ __forceinline__ float
norm(const float3& v)
{
	return sqrt(dot(v, v));
}

__device__ __forceinline__ float3
normalized(const float3& v)
{
	return v * rsqrt(dot(v, v));
}

__device__ __forceinline__ void computeRoots2(const float& b, const float& c, float3& roots)
{
	roots.x = 0.f;
	float d = b * b - 4.f * c;
	if (d < 0.f) 
		d = 0.f;
	float sd = sqrtf(d);
	roots.z = 0.5f * (-b + sd);
	roots.y = 0.5f * (-b - sd);
}

__device__ __forceinline__ void computeRoots3(float c0, float c1, float c2, float3& roots)
{
	if (c0 == 0)
	{
		computeRoots2(c2, c1, roots);
	}
	else
	{
		const float s_inv3 = 1.f / 3.f;
		const float s_sqrt3 = sqrtf(3.f);
		float c2_over_3 = c2 * s_inv3;
		float a_over_3 = (c1 - c2 * c2_over_3) * s_inv3;
		if (a_over_3 > 0.f)
			a_over_3 = 0.f;

		float half_b = 0.5f * (c0 + c2_over_3 * (2.f * c2_over_3 * c2_over_3 - c1));

		float q = half_b * half_b + a_over_3 * a_over_3 * a_over_3;
		if (q > 0.f)
			q = 0.f;
		float rho = sqrtf(-a_over_3);
		float theta = atan2(sqrtf(-q), half_b) * s_inv3;
		float cos_theta = __cosf(theta);
		float sin_theta = __sinf(theta);
		roots.x = c2_over_3 + 2.f * rho * cos_theta;
		roots.y = c2_over_3 - rho * (cos_theta + s_sqrt3 * sin_theta);
		roots.z = c2_over_3 - rho * (cos_theta - s_sqrt3 * sin_theta);
		//将得到的根按大小排列，x最小，y最大
		if (roots.x >= roots.y)
			dswap(roots.x, roots.y);

		if (roots.y >= roots.z)
		{
			dswap(roots.y, roots.z);

			if (roots.x >= roots.y) {
				dswap(roots.x, roots.y);
			}
		}
		if (roots.x <= 0) // 对称正半定矩阵的本征值不能是负的！将其设置为0
			computeRoots2(c2, c1, roots);
	}

}

//计算三维向量
struct Eigen33
{
public:
	template<int Rows>
	struct MiniMat
	{
		float3 data[Rows];
		__device__ __host__ __forceinline__ float3& operator[](int i) { return data[i]; }
		__device__ __host__ __forceinline__ const float3& operator[](int i) const { return data[i]; }
	};
	using Mat33 = MiniMat<3>;
	using Mat43 = MiniMat<4>;

	//用于计算给定向量垂直的单位向量
	static __forceinline__ __device__ float3 unitOrthogonal(const float3& src)
	{
		float3 perp;
		//x!=0 || y!=0
		if (!(src.x == 0) || !(src.y == 0))
		{
			float invnm = rsqrtf(src.x * src.x + src.y * src.y);
			perp.x = -src.y * invnm;
			perp.y = src.x * invnm;
			perp.z = 0.0f;
		}
		// x==0&&y==0
		else
		{
			float invnm = rsqrtf(src.z * src.z + src.y * src.y);
			perp.x = 0.0f;
			perp.y = -src.z * invnm;
			perp.z = src.y * invnm;
		}

		return perp;
	}

	__device__ __forceinline__ Eigen33(volatile float* mat_pkg_arg) : mat_pkg(mat_pkg_arg) {}

	__device__ __forceinline__ void compute(Mat33& tmp, Mat33& vec_tmp, Mat33& evecs, float3& evals)
	{

		float max01 = fmaxf(std::abs(mat_pkg[0]), std::abs(mat_pkg[1]));
		float max23 = fmaxf(std::abs(mat_pkg[2]), std::abs(mat_pkg[3]));
		float max45 = fmaxf(std::abs(mat_pkg[4]), std::abs(mat_pkg[5]));
		float m0123 = fmaxf(max01, max23);
		float scale = fmaxf(max45, m0123);

		if (scale <= 0)
			scale = 1.f;

		mat_pkg[0] /= scale;
		mat_pkg[1] /= scale;
		mat_pkg[2] /= scale;
		mat_pkg[3] /= scale;
		mat_pkg[4] /= scale;
		mat_pkg[5] /= scale;

		float c0 = m00() * m11() * m22()
			+ 2.f * m01() * m02() * m12()
			- m00() * m12() * m12()
			- m11() * m02() * m02()
			- m22() * m01() * m01();
		float c1 = m00() * m11() -
			m01() * m01() +
			m00() * m22() -
			m02() * m02() +
			m11() * m22() -
			m12() * m12();
		float c2 = m00() + m11() + m22();
		// x^3 - c2*x^2 + c1*x - c0 = 0
		computeRoots3(c0, c1, c2, evals);

		//最大值和最小值相等  以下部分已被演算过  正确
		if (evals.x == evals.z)
		{
			evecs[0] = make_float3(1.f, 0.f, 0.f);
			evecs[1] = make_float3(0.f, 1.f, 0.f);
			evecs[2] = make_float3(0.f, 0.f, 1.f);
		}
		//两个最小值相等
		else if (evals.x == evals.y)
		{
			// first and second equal                
			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot(vec_tmp[0], vec_tmp[0]);
			float len2 = dot(vec_tmp[1], vec_tmp[1]);
			float len3 = dot(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				evecs[2] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				evecs[2] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				evecs[2] = vec_tmp[2] * rsqrtf(len3);
			}

			evecs[1] = unitOrthogonal(evecs[2]);
			evecs[0] = cross(evecs[1], evecs[2]);

		}
		//两个最大值相等
		else if (evals.z == evals.y)
		{
			// second and third equal                                    
			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();

			tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot(vec_tmp[0], vec_tmp[0]);
			float len2 = dot(vec_tmp[1], vec_tmp[1]);
			float len3 = dot(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				evecs[0] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				evecs[0] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				evecs[0] = vec_tmp[2] * rsqrtf(len3);
			}

			evecs[1] = unitOrthogonal(evecs[0]);
			evecs[2] = cross(evecs[0], evecs[1]);
		}
		//三个不同的特征值
		else
		{

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.z; tmp[1].y -= evals.z; tmp[2].z -= evals.z;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			float len1 = dot(vec_tmp[0], vec_tmp[0]);
			float len2 = dot(vec_tmp[1], vec_tmp[1]);
			float len3 = dot(vec_tmp[2], vec_tmp[2]);

			float mmax[3];

			unsigned int min_el = 2;
			unsigned int max_el = 2;
			if (len1 >= len2 && len1 >= len3)
			{
				mmax[2] = len1;
				evecs[2] = vec_tmp[0] * rsqrtf(len1);
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[2] = len2;
				evecs[2] = vec_tmp[1] * rsqrtf(len2);
			}
			else
			{
				mmax[2] = len3;
				evecs[2] = vec_tmp[2] * rsqrtf(len3);
			}

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.y; tmp[1].y -= evals.y; tmp[2].z -= evals.y;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			len1 = dot(vec_tmp[0], vec_tmp[0]);
			len2 = dot(vec_tmp[1], vec_tmp[1]);
			len3 = dot(vec_tmp[2], vec_tmp[2]);

			if (len1 >= len2 && len1 >= len3)
			{
				mmax[1] = len1;
				evecs[1] = vec_tmp[0] * rsqrtf(len1);
				min_el = len1 <= mmax[min_el] ? 1 : min_el;
				max_el = len1 > mmax[max_el] ? 1 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[1] = len2;
				evecs[1] = vec_tmp[1] * rsqrtf(len2);
				min_el = len2 <= mmax[min_el] ? 1 : min_el;
				max_el = len2 > mmax[max_el] ? 1 : max_el;
			}
			else
			{
				mmax[1] = len3;
				evecs[1] = vec_tmp[2] * rsqrtf(len3);
				min_el = len3 <= mmax[min_el] ? 1 : min_el;
				max_el = len3 > mmax[max_el] ? 1 : max_el;
			}

			tmp[0] = row0();  tmp[1] = row1();  tmp[2] = row2();
			tmp[0].x -= evals.x; tmp[1].y -= evals.x; tmp[2].z -= evals.x;

			vec_tmp[0] = cross(tmp[0], tmp[1]);
			vec_tmp[1] = cross(tmp[0], tmp[2]);
			vec_tmp[2] = cross(tmp[1], tmp[2]);

			len1 = dot(vec_tmp[0], vec_tmp[0]);
			len2 = dot(vec_tmp[1], vec_tmp[1]);
			len3 = dot(vec_tmp[2], vec_tmp[2]);


			if (len1 >= len2 && len1 >= len3)
			{
				mmax[0] = len1;
				evecs[0] = vec_tmp[0] * rsqrtf(len1);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}
			else if (len2 >= len1 && len2 >= len3)
			{
				mmax[0] = len2;
				evecs[0] = vec_tmp[1] * rsqrtf(len2);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}
			else
			{
				mmax[0] = len3;
				evecs[0] = vec_tmp[2] * rsqrtf(len3);
				min_el = len3 <= mmax[min_el] ? 0 : min_el;
				max_el = len3 > mmax[max_el] ? 0 : max_el;
			}

			unsigned mid_el = 3 - min_el - max_el;
			evecs[min_el] = normalized(cross(evecs[(min_el + 1) % 3], evecs[(min_el + 2) % 3]));
			evecs[mid_el] = normalized(cross(evecs[(mid_el + 1) % 3], evecs[(mid_el + 2) % 3]));

		}
		evals *= scale;

	}
private:
	volatile float* mat_pkg;

	__device__  __forceinline__ float m00() const { return mat_pkg[0]; }
	__device__  __forceinline__ float m01() const { return mat_pkg[1]; }
	__device__  __forceinline__ float m02() const { return mat_pkg[2]; }
	__device__  __forceinline__ float m10() const { return mat_pkg[1]; }
	__device__  __forceinline__ float m11() const { return mat_pkg[3]; }
	__device__  __forceinline__ float m12() const { return mat_pkg[4]; }
	__device__  __forceinline__ float m20() const { return mat_pkg[2]; }
	__device__  __forceinline__ float m21() const { return mat_pkg[4]; }
	__device__  __forceinline__ float m22() const { return mat_pkg[5]; }

	__device__  __forceinline__ float3 row0() const { return make_float3(m00(), m01(), m02()); }
	__device__  __forceinline__ float3 row1() const { return make_float3(m10(), m11(), m12()); }
	__device__  __forceinline__ float3 row2() const { return make_float3(m20(), m21(), m22()); }

};

//矩阵运算
__device__  __forceinline__ void asDiagonal(const float* a, const float* b, float *result,int n,int nr) {
	for (int i = 0; i < nr; i++)
		for (int j = 0; j < n; j++)
			result[i * maxresult + j] = a[i * maxresult + j] * b[j];
}

//矩阵乘法
__device__  __forceinline__ void MatirxCross(const float* a, const float* b, float* result, int n, int nr) {
	for (int i = 0; i < nr; i++)
		for (int j = 0; j < nr; j++)
			for(int k = 0; k < n; k++)
				result[i * nr + j] += a[i * maxresult + k] * b[j * maxresult + k];
}

//矩阵向量乘法
__device__  __forceinline__ void MatrixVectorCross(const float* a, const float* b, float* result, int n, int nr) {
for (int i = 0; i < nr; i++)
	for (int j = 0; j < n; j++)
			result[i] += a[i * maxresult + j] * b[j];
}

__device__  __forceinline__ bool inverseMatrix(float* matrix, float* inverse, int n) {

	for (size_t i = 0; i < n; ++i) {
		inverse[i * n + i] = 1.0;
	}
	for (size_t i = 0; i < n; ++i) {
		if (matrix[i * n + i] == 0.0) {
			return false;
		}

		float diagonalRecip = 1.0 / matrix[i * n + i];
		for (size_t j = 0; j < n; ++j) {
			matrix[i * n + j] = matrix[i * n + j] * diagonalRecip;
			inverse[i * n + j] = inverse[i * n + j] * diagonalRecip;
		}

		for (size_t j = 0; j < n; ++j) {
			if (i != j) {
				float scale = matrix[j * n + i];
				for (size_t k = 0; k < n; ++k) {
					matrix[j * n + k] = matrix[j * n + k] - scale * matrix[i * n + k];
					inverse[j * n + k] = inverse[j * n + k] - scale * inverse[i * n + k];
				}
			}
		}

	}
	return true;
}

//逆矩阵向量乘法
__device__  __forceinline__ void InverseMatrixVectorCross(const float* a, const float* b, float* result, int nr) {
	for (int i = 0; i < nr; i++)
		for (int j = 0; j < nr; j++)
			result[i] += a[i * nr + j] * b[j];
}

//LU分解
__device__  __forceinline__ bool luDecomposition(const float* A,const float* b, float* L, float* U, int n) {
	for (int i = 0; i < n; i++) {
		for (int k = i; k < n; k++) {
			double sum = 0;
			for (int j = 0; j < i; j++) {
				sum += (L[i * n + j] * U[j * n + k]);
			}
			U[i * n + k] = A[i * n + k] - sum;
		}

		for (int k = i; k < n; k++) {
			if (i == k) {
				L[i * n + i] = 1;
			}
			else {
				double sum = 0;
				for (int j = 0; j < i; j++) {
					sum += (L[k * n + j] * U[j * n + i]);
				}
				L[k * n + i] = (A[k * n + i] - sum) / U[i * n + i];
			}
		}
	}
	return true;
}

//前向替代计算y
__device__  __forceinline__ void comex(float* b, float* L, float* U, float* y, float* x, int n) {
	for (int i = 0; i < n; i++) {
		double sum = 0;
		for (int j = 0; j < i; j++) {
			sum += L[i * n + j] * y[j];
		}
		y[i] = b[i] - sum;
	}

	//后向替代计算x
	for (int i = n - 1; i >= 0; i--) {
		double sum = 0;
		for (int j = i + 1; j < n; j++) {
			sum += U[i * n + j] * x[j];
		}
		x[i] = (y[i] - sum) / U[i * n + i];
	}
}
//正定矩阵
const int nr_coeff = (order + 1) * (order + 2) / 2;
const int Matirxn = nr_coeff * (nr_coeff + 1) / 2;
class SymmetricMatrix {
public:
	__device__  __forceinline__ SymmetricMatrix(float *P_weight_Pt,int n) {
		n_ = n;
		for (int i = 0; i < n_; i++)
			for (int j = i; j < n_; j++)
				data_[i * n - i * (i - 1) / 2 + j - i] = P_weight_Pt[i*n+j];
	}

	__device__  __forceinline__ float& operator()(int i, int j) {
		if (i > j) dswap(i, j);
		return data_[i * n_ - i * (i - 1) / 2 + j - i];
	}

	__device__  __forceinline__ const float& operator()(int i, int j) const {
		if (i > j) dswap(i, j);
		return data_[i * n_- i * (i - 1) / 2 + j - i];
	}

	__device__  __forceinline__ int size() const { return n_; }

private:
	int n_;
	float data_[Matirxn];
};
__device__  __forceinline__ bool llt(SymmetricMatrix& A) {
	int n = A.size();
	for (int k = 0; k < n; ++k) {
		if (A(k, k) <= 0) {
			//printf("Matrix is not positive definite.\n" );
			return false;
		}
		A(k, k) = sqrt(A(k, k));
		for (int i = k + 1; i < n; ++i) {
			A(i, k) /= A(k, k);
		}

		for (int i = k + 1; i < n; ++i) {
			for (int j = k + 1; j <= i; ++j) {
				A(i, j) -= A(i, k) * A(j, k);
			}
		}
	}
	return true;
}

__device__  __forceinline__ void solveInPlace(const SymmetricMatrix& A, float* b) {
	int n = A.size();
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < i; ++j) {
			b[i] -= A(i, j) * b[j];
		}
		b[i] /= A(i, i);
	}
	for (int i = n - 1; i >= 0; --i) {
		for (int j = i + 1; j < n; ++j) {
			b[i] -= A(j, i) * b[j];  // 注意A(j, i)等同于A(i, j)
		}
		b[i] /= A(i, i);
	}
}

//比重
__device__ __forceinline__ float weight_func(const float sq_dist,const float search_radius) { 
	return std::exp(-sq_dist / search_radius / search_radius); 
};
//排序函数
template<typename V, typename K>
__device__ __forceinline__ void bitonicSortWarp(volatile K* keys, volatile V* vals, unsigned int dir = 1)
{
	const unsigned int arrayLength = 64;
	unsigned int lane = threadIdx.x & 31;
	//将序列变成双调序列
	for (unsigned int size = 2; size < arrayLength; size <<= 1)
	{
		//Bitonic merge   size是2的倍数，所以size只能是10**0 
		//(lane & (size / 2)) != 0用来判断是在lane处理的线程在size前半部分还是后半部分
		//size的前半部分为递增序列，后半部分为递减序列
		unsigned int ddd = dir ^ ((lane & (size / 2)) != 0);

		for (unsigned int stride = size / 2; stride > 0; stride >>= 1)
		{
			unsigned int pos = 2 * lane - (lane & (stride - 1));

			if ((keys[pos] > keys[pos + stride]) == ddd)
			{
				dswap(keys[pos], keys[pos + stride]);
				dswap(vals[pos], vals[pos + stride]);
			}
		}
	}
	//对双调序列进行排序
	//ddd == dir for the last bitonic merge step
	for (unsigned int stride = arrayLength / 2; stride > 0; stride >>= 1)
	{
		unsigned int pos = 2 * lane - (lane & (stride - 1));

		if ((keys[pos] > keys[pos + stride]) == dir)
		{
			dswap(keys[pos], keys[pos + stride]);
			dswap(vals[pos], vals[pos + stride]);
		}
	}
}