#pragma once
#include <limits>
#include"Octree Device Iterator.hpp"
#include"Warp.hpp"
#include"Morton.hpp"
#include"Utils.hpp"
struct KBatch
{
	OctreeGlobal octree;
	int queries_num;
	const PointType* queries;
	mutable float3* normals;
	const float* points;
	int k;
	int points_step; // elem step
	mutable int* knnindices;
	mutable float* knndist;
};


struct Warp_knnSearch
{
public:
	using OctreeIterator = OctreeIteratorDeviceNS;

	const KBatch& batch;

	int query_index;//表示当前正在处理的查询点的编号
	float3 query;//表示当前正在处理的查询点的坐标

	float min_distance;
	OctreeIterator iterator;

	__device__ __forceinline__ Warp_knnSearch(const KBatch& batch_arg, const int query_index_arg)
		: batch(batch_arg), query_index(query_index_arg), min_distance(IN_MAX), iterator(batch.octree) {}

	__device__ __forceinline__ void launch(bool active)
	{
		__shared__ float covbuffer[6][32];
		const unsigned int laneId = Warp::laneId();
		for (int i = laneId; i < batch.queries_num * batch.k; i += 32)
			batch.knndist[i] = IN_MAX;
		if (active)
		{
			PointType q = batch.queries[query_index];
			query = make_float3(q.x, q.y, q.z);
		}
		else
			query_index = -1;
		while (__any_sync(0xFFFFFFFF, active))
		{
			int leaf = -1;
			if (active)
				leaf = examineNode(iterator);
			processLeaf(leaf);
			active = active && iterator.level >= 0;
		}
		int mask = __ballot_sync(0xFFFFFFFF, query_index != -1);
		while (mask) {
			const int active_lane = __ffs(mask) - 1; //[0..31]
		/*	if (threadIdx.x == 0&&active_lane==0) {
				for (int i = 0; i < 90; i++) {
					if (i != 0 && i % 9 == 0)printf("\n");
					printf("%f  ", batch.knndist[i]);
				}
				printf("\n");
			}*/
			mask &= ~(1 << active_lane);		    
			float3 centre = make_float3(0.f, 0.f, 0.f);
			float3 cpoint = make_float3(0.f, 0.f, 0.f);
			int offest = query_index * batch.k;
			const int otheroffest = __shfl_sync(0xFFFFFFFF, offest, active_lane);
			for (int i = laneId; i < batch.k; i += 32) {
				cpoint = cpoint + make_float3(batch.points[batch.knnindices[otheroffest + laneId]], batch.points[batch.knnindices[otheroffest + laneId] + batch.points_step], batch.points[batch.knnindices[otheroffest + laneId] + batch.points_step * 2]);
			}
			volatile float* buffer = &covbuffer[0][0];
			centre.x = Warp::reduce(buffer, cpoint.x);
			centre.y = Warp::reduce(buffer, cpoint.y);
			centre.z = Warp::reduce(buffer, cpoint.z);
			centre = centre / batch.k;
			float dx2 = 0, dxy = 0, dxz = 0, dy2 = 0, dyz = 0, dz2 = 0; float3 d;
			cpoint = make_float3(0, 0, 0);
			for (int i = laneId; i < batch.k; i += 32) {
				cpoint = cpoint + make_float3(batch.points[batch.knnindices[otheroffest + laneId]], batch.points[batch.knnindices[otheroffest + laneId] + batch.points_step], batch.points[batch.knnindices[otheroffest + laneId] + batch.points_step * 2]);
				d = cpoint - centre;
				dx2 += d.x * d.x;  dxy += d.x * d.y;
				dxz += d.x * d.z;  dy2 += d.y * d.y;
				dyz += d.y * d.z;  dz2 += d.z * d.z;
			}
			float a =Warp::reduce(&covbuffer[0][0], dx2);
			float b =Warp::reduce(&covbuffer[1][0], dxy);
			float c =Warp::reduce(&covbuffer[2][0], dxz);
			float g =Warp::reduce(&covbuffer[3][0], dy2);
			float e =Warp::reduce(&covbuffer[4][0], dyz);
			float f =Warp::reduce(&covbuffer[5][0], dz2);
			if (laneId == active_lane)
			{
				volatile float* cov = &covbuffer[0][0];
				cov[0] = covbuffer[0][0];
				cov[1] = covbuffer[1][0];
				cov[2] = covbuffer[2][0];
				cov[3] = covbuffer[3][0];
				cov[4] = covbuffer[4][0];
				cov[5] = covbuffer[5][0];
				using Mat33 = Eigen33::Mat33;
				Eigen33 eigen33(&cov[0]);
				Mat33& tmp = (Mat33&)covbuffer[1][0];
				Mat33& vec_tmp = (Mat33&)covbuffer[2][0];
				Mat33& evecs = (Mat33&)covbuffer[3][0];
				float3 evals;
				eigen33.compute(tmp, vec_tmp, evecs, evals);
				batch.normals[query_index] = evecs[0];
			}
		}
	}
private:
	__device__ __forceinline__ int examineNode(OctreeIterator& iterator)
	{
		const int node_idx = *iterator;
		const int code = batch.octree.codes[node_idx];

		float3 node_minp = batch.octree.minp;
		float3 node_maxp = batch.octree.maxp;
		calcBoundingBox(iterator.level, code, node_minp, node_maxp);

		if (checkIfNodeOutsideSphere(node_minp, node_maxp, query, min_distance))
		{
			++iterator;
			return -1;
		}

		const int node = batch.octree.nodes[node_idx];
		const int children_mask = node & 0xFF;
		const bool isLeaf = children_mask == 0;
		if (isLeaf)
		{
			++iterator;
			return node_idx;
		}
		const int first = node >> 8;
		const int len = __popc(children_mask);
		iterator.gotoNextLevel(first, len);
		return -1;
	};

	__device__ __forceinline__ void processLeaf(const int node_idx)
	{
		int mask = __ballot_sync(0xFFFFFFFF, node_idx != -1);
		const unsigned int laneId = Warp::laneId();
		__shared__ float distbuffer[64]; __shared__ int idxbuffer[64];
		while (mask)
		{
			distbuffer[laneId] = IN_MAX; distbuffer[laneId + 32] = IN_MAX;
			const int active_lane = __ffs(mask) - 1; //[0..31]
			mask &= ~(1 << active_lane);
			int fbeg, fend;
			if (active_lane == laneId)
			{
				fbeg = batch.octree.begs[node_idx];
				fend = batch.octree.ends[node_idx];
			}
			const int beg = __shfl_sync(0xFFFFFFFF, fbeg, active_lane);
			const int end = __shfl_sync(0xFFFFFFFF, fend, active_lane);
			const float3 active_query = make_float3(
				__shfl_sync(0xFFFFFFFF, query.x, active_lane),
				__shfl_sync(0xFFFFFFFF, query.y, active_lane),
				__shfl_sync(0xFFFFFFFF, query.z, active_lane)
			);
			int offest = query_index * batch.k;
			const int otheroffest = __shfl_sync(0xFFFFFFFF, offest, active_lane);
			volatile float* Dbuffer = &distbuffer[0]; volatile int* Ibuffer = &idxbuffer[0];
			int count = 0; 
			int iters = (end-beg+31)/32;
			for (int idx = beg + laneId, it = 0; it < iters; idx += 32, it++, count++) {
				if (idx < end) {
					const float dx = batch.points[idx] - active_query.x;
					const float dy = batch.points[idx + batch.points_step] - active_query.y;
					const float dz = batch.points[idx + batch.points_step * 2] - active_query.z;
					const float d2 = dx * dx + dy * dy + dz * dz;
					if (count > 1)count = 0;
					distbuffer[laneId + count * 32] = d2;
					idxbuffer[laneId + count * 32] = idx;
				}
				if (count == 1) {
					bitonicSortWarp(Dbuffer, Ibuffer);
					if (laneId < batch.k) {
						distbuffer[batch.k + laneId] = batch.knndist[otheroffest + laneId];
						idxbuffer[batch.k + laneId] = batch.knnindices[otheroffest + laneId];
					}
					bitonicSortWarp(Dbuffer, Ibuffer);
					if (laneId < batch.k) {
						batch.knndist[otheroffest + laneId] = distbuffer[laneId];
						batch.knnindices[otheroffest + laneId] = idxbuffer[laneId];
						distbuffer[laneId] = IN_MAX;
					}
				}
			}
			if (count != 2) {
				bitonicSortWarp(Dbuffer, Ibuffer);
				if (laneId < batch.k) {
					distbuffer[batch.k + laneId] = batch.knndist[otheroffest + laneId];
					idxbuffer[batch.k + laneId] = batch.knnindices[otheroffest + laneId];
				}
				bitonicSortWarp(Dbuffer, Ibuffer);
				if (laneId < batch.k) {
					batch.knndist[otheroffest + laneId] = distbuffer[laneId];
					batch.knnindices[otheroffest + laneId] = idxbuffer[laneId];
				}
			}
			if (laneId == active_lane) {
				min_distance = sqrt(batch.knndist[offest + batch.k - 1]);
			}
		}
	}
};


