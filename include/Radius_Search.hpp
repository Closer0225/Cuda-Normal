#pragma once
#include"Octree Device Iterator.hpp"
#include"Warp.hpp"
#include"Morton.hpp"
#include"Utils.hpp"
struct Batch
{
    OctreeGlobal octree;
    int queries_num;
    mutable PointType* queries;
    mutable float3* normals;
    PtrStep<float> points;
    int max_results;
    float radiuses;
};

struct KernelPolicy
{
    enum
    {
        CTA_SIZE = 32,

        MAX_LEVELS_PLUS_ROOT = 11,

        CHECK_FLAG = 1 << 31
    };

};

__shared__ volatile int cta_buffer[KernelPolicy::CTA_SIZE];


struct Warp_radiusSearch
{
public:
    using OctreeIterator = OctreeIteratorDeviceNS;

    const Batch& batch;
    OctreeIterator iterator;
    int output[maxresult];
    int found_count;
    int query_index;
    float3 query;
    float radius;

    __device__ __forceinline__ Warp_radiusSearch(const Batch& batch_arg, const int query_index_arg)
        : batch(batch_arg), iterator(batch.octree), found_count(0), query_index(query_index_arg) {}

    __device__ __forceinline__ void launch(bool active)
    {
        const unsigned int laneId = Warp::laneId();
        if (active)
        {
            PointType q = batch.queries[query_index];
            query = make_float3(q.x, q.y, q.z);
            radius = batch.radiuses;
        }
        else
            query_index = -1;

        while (__any_sync(0xFFFFFFFF, active))
        {
            int leaf = -1;

            if (active)
                leaf = examineNode(iterator);

            processLeaf(leaf);

            active = active && iterator.level >= 0 && found_count < batch.max_results;
        }
        if (query_index != -1&& found_count>=3) {
            float3 centre = make_float3(0.f, 0.f, 0.f);
            float3 dpoint = make_float3(0.f, 0.f, 0.f);
            for (int i = 0; i < found_count; i++) {
                dpoint = dpoint + make_float3(batch.points.ptr(0)[output[i]], batch.points.ptr(1)[output[i]], batch.points.ptr(2)[output[i]]);
            }
            centre = dpoint / found_count;
            float dx2 = 0, dxy = 0, dxz = 0, dy2 = 0, dyz = 0, dz2 = 0;
            float3 d;
            dpoint = make_float3(0, 0, 0);
            for (int i = 0; i < found_count; i++) {
                dpoint = make_float3(batch.points.ptr(0)[output[i]], batch.points.ptr(1)[output[i]], batch.points.ptr(2)[output[i]]);
                d = dpoint - centre;
                dx2 += d.x * d.x;  dxy += d.x * d.y;
                dxz += d.x * d.z;  dy2 += d.y * d.y;
                dyz += d.y * d.z;  dz2 += d.z * d.z;
            }
            float covbuffer[36];
            volatile float* cov = &covbuffer[0];
            cov[0] = dx2;
            cov[1] = dxy;
            cov[2] = dxz;
            cov[3] = dy2;
            cov[4] = dyz;
            cov[5] = dz2;
            using Mat33 = Eigen33::Mat33;
            Eigen33 eigen33(&cov[0]);
            Mat33& tmp = (Mat33&)covbuffer[9];
            Mat33& vec_tmp = (Mat33&)covbuffer[18];
            Mat33& evecs = (Mat33&)covbuffer[27];
            float3 evals;
            eigen33.compute(tmp, vec_tmp, evecs, evals);
            batch.normals[query_index] = evecs[0];
            float3 normal = batch.normals[query_index];
            if (length(normal) > 0.5f)
            {
                float3 q;
                q = make_float3(batch.queries[query_index].x, batch.queries[query_index].y, batch.queries[query_index].z);
                query = query + normal * dot(normal, centre - q);
            }
            if (order > 1)
            {
                float3 plane_normal;
                plane_normal.x = normal.x;
                plane_normal.y = normal.y;
                plane_normal.z = normal.z;
                auto v_axis = Eigen33::unitOrthogonal(plane_normal);
                auto u_axis = cross(plane_normal, v_axis);
                float search_radius = batch.radiuses;

                auto num_neighbors = found_count;
                if (order > 1)
                {
                    const int nr_coeff = (order + 1) * (order + 2) / 2;

                    if (num_neighbors >= nr_coeff)
                    {
                        float weight_vec[maxresult]  ;
                        float P[nr_coeff * maxresult];
                        float f_vec[maxresult];
                        float P_weight_Pt[nr_coeff * nr_coeff] = { 0 };
                        float3 de_meaned[maxresult];
                        for (std::size_t ni = 0; ni < num_neighbors; ++ni)
                        {
                            de_meaned[ni].x = batch.points.ptr(0)[output[ni]] - query.x;//mean定义在computeNormal中的center
                            de_meaned[ni].y = batch.points.ptr(1)[output[ni]] - query.y;
                            de_meaned[ni].z = batch.points.ptr(2)[output[ni]] - query.z;
                            weight_vec[ni] = weight_func(dot(de_meaned[ni], de_meaned[ni]), search_radius);//移植一个weight_func()函数
                        }
                        //遍历邻居，在局部坐标系中转换它们，保存高度和多项式项的值
                        for (std::size_t ni = 0; ni < num_neighbors; ++ni)
                        {
                            //转换坐标
                            const float u_coord = dot(de_meaned[ni], u_axis);
                            const float v_coord = dot(de_meaned[ni], v_axis);
                            f_vec[ni] = dot(de_meaned[ni], plane_normal);
                            //计算多项式在当前点的项
                            int j = 0;
                            float u_pow = 1;
                            for (int ui = 0; ui <= order; ++ui)
                            {
                                float v_pow = 1;
                                for (int vi = 0; vi <= order - ui; ++vi, j++)
                                {
                                    P[j * maxresult + ni] = u_pow * v_pow;
                                    v_pow *= v_coord;
                                }
                                u_pow *= u_coord;
                            }
                        }
                        //计算系数
                        float P_weight[nr_coeff * maxresult];
                        asDiagonal(P, weight_vec, P_weight, num_neighbors, nr_coeff);
                        MatirxCross(P_weight, P, P_weight_Pt, num_neighbors, nr_coeff);
                        float c_vec[nr_coeff] = { 0 };
                        MatrixVectorCross(P_weight, f_vec, c_vec, num_neighbors, nr_coeff);
                        //使用LLT分解（Cholesky分解）来求解多项式系数 c_vec
                        SymmetricMatrix A(P_weight_Pt, nr_coeff);
                        bool isllt = llt(A);
                        if (isllt) {
                            solveInPlace(A, c_vec);
                            //simple方法
                            query = query + (normal * c_vec[0]);
                            float3 proj_normal;
                            proj_normal = plane_normal - u_axis * c_vec[order + 1] - v_axis * c_vec[1];
							proj_normal=normalized(proj_normal);
                            normal.x = proj_normal.x;
                            normal.y = proj_normal.y;
                            normal.z = proj_normal.z;
                        }
                    }
                } 
            }
            batch.normals[query_index] = normal;
            batch.queries[query_index] = make_float4(query.x, query.y, query.z, 0);
        }
        else
        batch.normals[query_index] = make_float3(0.0,0.0,0.0);
    }
private:
    __device__ __forceinline__ int examineNode(OctreeIterator& iterator)
    {
        const int node_idx = *iterator;
        const int code = batch.octree.codes[node_idx];

        float3 node_minp = batch.octree.minp;
        float3 node_maxp = batch.octree.maxp;
        calcBoundingBox(iterator.level, code, node_minp, node_maxp);


        if (checkIfNodeOutsideSphere(node_minp, node_maxp, query, radius))
        {
            ++iterator;
            return -1;
        }

        if (checkIfNodeInsideSphere(node_minp, node_maxp, query, radius))
        {
            ++iterator;
            return node_idx;
        }


        const int node = batch.octree.nodes[node_idx];
        const int children_mask = node & 0xFF;
        const bool isLeaf = children_mask == 0;

        if (isLeaf)
        {
            ++iterator;
            //将最高位设置为 1，以此作为标记表示这个节点是一个叶子节点
            return (node_idx | KernelPolicy::CHECK_FLAG);
        }

        const int first = node >> 8;
        const int len = __popc(children_mask);
        iterator.gotoNextLevel(first, len);
        return -1;
    };

    __device__ __forceinline__ void processLeaf(int leaf)
    {
        int mask = __ballot_sync(0xFFFFFFFF, leaf != -1);
        __shared__ int OutPut[maxresult];
        while (mask)
        {
            const unsigned int laneId = Warp::laneId();
            int active_lane = __ffs(mask) - 1; //[0..31]

            mask &= ~(1 << active_lane);

            const int active_found_count = __shfl_sync(0xFFFFFFFF, found_count, active_lane);
            //将标记的叶子节点首位变成0
            const int node_idx = leaf & ~KernelPolicy::CHECK_FLAG;

            int fbeg, fend;
            if (active_lane == laneId)
            {
                fbeg = batch.octree.begs[node_idx];
                fend = batch.octree.ends[node_idx];
            }
            const int beg = __shfl_sync(0xFFFFFFFF, fbeg, active_lane);
            const int end = __shfl_sync(0xFFFFFFFF, fend, active_lane);

            const int active_query_index = __shfl_sync(0xFFFFFFFF, query_index, active_lane);

            int length = end - beg;

            int* out = &OutPut[0];
            const int length_left = batch.max_results - active_found_count;

            //leaf& KernelPolicy::CHECK_FLAG判断是否为叶节点 leaf的首位必定为0，node只用了16位
            const int test = __any_sync(0xFFFFFFFF, active_lane == laneId && (leaf & KernelPolicy::CHECK_FLAG));

            if (test)
            {
                const float radius2 = __shfl_sync(0xFFFFFFFF, radius * radius, active_lane);

                const float3 active_query = make_float3(
                    __shfl_sync(0xFFFFFFFF, query.x, active_lane),
                    __shfl_sync(0xFFFFFFFF, query.y, active_lane),
                    __shfl_sync(0xFFFFFFFF, query.z, active_lane)
                );

                length = TestWarpKernel(beg, active_query, radius2, length, out, length_left);
            }
            else
            {
                length = min(length, length_left);
                for (int i = laneId; i < length; i += 32)
                    out[i] = beg + i;
            }

            if (active_lane == laneId)
            {
                for (int i = 0; i < length; i++)
                    output[found_count + i] = OutPut[i];
                found_count += length;
            }
        }
    }

    __device__ __forceinline__ int TestWarpKernel(const int beg, const float3& active_query, const float radius2, const int length, int* out, const int length_left)
    {
        unsigned int idx = Warp::laneId();
        //const int last_threadIdx = threadIdx.x - idx + 31;

        int total_new = 0;

        for (;;)
        {
            int take = 0;

            if (idx < length)
            {
                const float dx = batch.points.ptr(0)[beg + idx] - active_query.x;
                const float dy = batch.points.ptr(1)[beg + idx] - active_query.y;
                const float dz = batch.points.ptr(2)[beg + idx] - active_query.z;

                const float d2 = dx * dx + dy * dy + dz * dz;

                if (d2 < radius2)
                    take = 1;
            }

            cta_buffer[threadIdx.x] = take;

            const int offset = scan_warp<exclusive>(cta_buffer);

            const bool out_of_bounds = (offset + total_new) >= length_left;

            if (take && !out_of_bounds)
                out[offset] = beg + idx;

            const int new_nodes = cta_buffer[31];

            idx += 32;

            total_new += new_nodes;
            out += new_nodes;

            if (__all_sync(0xFFFFFFFF, idx >= length) || __any_sync(0xFFFFFFFF, out_of_bounds) || total_new == length_left)
                break;
        }
        return min(total_new, length_left);
    }
};



