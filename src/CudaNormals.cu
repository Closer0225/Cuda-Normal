#include"Knn_Search.hpp"
#include"Radius_Search.hpp"
#include <thrust/device_ptr.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include"Clock.h"
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>
//八叉树
const static int max_points_per_leaf = 96;
enum
{
    GRID_SIZE = 1,//1个网格
    CTA_SIZE = 1024 - 32,//块的大小
    STRIDE = CTA_SIZE,//块的大小决定步长
    LEVEL_BITS_NUM = 3,
    ARITY = 1 << LEVEL_BITS_NUM//每个节点最多8个孩子
};
__shared__ int nodes_num;
__shared__ int tasks_beg;
__shared__ int tasks_end;
__shared__ int total_new;
__shared__ volatile int offsets[CTA_SIZE];
template<typename PointType>
struct SelectMinPoint
{
    __host__ __device__ __forceinline__ PointType operator()(const PointType& e1, const PointType& e2) const
    {
        PointType result;
        result.x = fmin(e1.x, e2.x);
        result.y = fmin(e1.y, e2.y);
        result.z = fmin(e1.z, e2.z);
        return result;
    }
};

template<typename PointType>
struct SelectMaxPoint
{
    __host__ __device__ __forceinline__ PointType operator()(const PointType& e1, const PointType& e2) const
    {
        PointType result;
        result.x = fmax(e1.x, e2.x);
        result.y = fmax(e1.y, e2.y);
        result.z = fmax(e1.z, e2.z);
        return result;
    }
};
template<typename PointType>
struct PointType_to_tuple
{
    __device__ __forceinline__ thrust::tuple<float, float, float> operator()(const PointType& arg) const
    {
        thrust::tuple<float, float, float> res;
        res.get<0>() = arg.x;
        res.get<1>() = arg.y;
        res.get<2>() = arg.z;
        return res;
    }
};
struct SingleStepBuild
{
    const int* codes;
    int points_number;
    mutable OctreeGlobal octree;

    static __device__ __forceinline__ int divUp(int total, int grain) { return (total + grain - 1) / grain; };

    __device__ __forceinline__ int FindCells(int task, int level, int cell_begs[], char cell_code[]) const
    {
        int cell_count = 0;

        int beg = octree.begs[task];
        int end = octree.ends[task];

        if (end - beg < max_points_per_leaf)
        {
            //cell_count == 0;
        }
        else
        {
            //0<=cur_code<=7 不同层级移位，找到不同子节点的孩子
            int cur_code = Morton::extractLevelCode(codes[beg], level);

            cell_begs[cell_count] = beg;
            cell_code[cell_count] = cur_code;
            ++cell_count;

            int last_code = Morton::extractLevelCode(codes[end - 1], level);
            if (last_code == cur_code)
            {
                cell_begs[cell_count] = end;
            }
            else
            {
                for (;;)
                {
                    int search_code = cur_code + 1;
                    if (search_code == 8)
                    {
                        cell_begs[cell_count] = end;
                        break;
                    }
                    //找到这个节点共有多少个点
                    int morton_code = Morton::shiftLevelCode(search_code, level);
                    //pos是大于该节点的第一个节点(是search_code的节点)，因为search_code刚好是该节点的下一个节点
                    //在CompareByLevelCode方法中 search_code的shiftLevelCode移level位结果与extractLevelCode的移level位结果刚好使得该子节点编号与另一个codes中的extractLevelCode的移level位结果相对比，比较codes[i]是否为该子节点
                    int pos = Lower_bound(codes + beg, codes + end, morton_code, CompareByLevelCode(level)) - codes;

                    if (pos == end)
                    {
                        cell_begs[cell_count] = end;
                        break;
                    }
                    cur_code = Morton::extractLevelCode(codes[pos], level);

                    cell_begs[cell_count] = pos;
                    cell_code[cell_count] = cur_code;
                    ++cell_count;
                    beg = pos;
                }
            }
        }
        return cell_count;
    }


    __device__  __forceinline__ void build() const
    {
        static_assert((max_points_per_leaf % 32) == 0, "max_points_per_leaf must be a multiple of 32");

        if (threadIdx.x == 0)
        {
            //init root
            octree.codes[0] = 0;
            octree.nodes[0] = 0;
            octree.begs[0] = 0;
            octree.ends[0] = points_number;
            octree.parent[0] = -1;

            nodes_num = 1;
            tasks_beg = 0;
            tasks_end = 1;
            total_new = 0;
        }

        int level = 0;

        int  cell_begs[ARITY + 1];
        char cell_code[ARITY];

        __syncthreads();

        while (tasks_beg < tasks_end && level < Morton::levels)
        {
            int task_count = tasks_end - tasks_beg;
            int iters = divUp(task_count, CTA_SIZE);

            int task = tasks_beg + threadIdx.x;


            for (int it = 0; it < iters; ++it, task += STRIDE)
            {
                int cell_count = (task < tasks_end) ? FindCells(task, level, cell_begs, cell_code) : 0;

                offsets[threadIdx.x] = cell_count;
                __syncthreads();

                scan_block<exclusive>(offsets);


                if (task < tasks_end)
                {
                    if (cell_count > 0)
                    {
                        int parent_code_shifted = octree.codes[task] << LEVEL_BITS_NUM;
                        int offset = nodes_num + offsets[threadIdx.x];

                        int mask = 0;
                        for (int i = 0; i < cell_count; ++i)
                        {
                            octree.begs[offset + i] = cell_begs[i];
                            octree.ends[offset + i] = cell_begs[i + 1];
                            //每3个为一层，共10层，父节点为parent_code_shifted层，儿子节点为cell_code层
                            octree.codes[offset + i] = parent_code_shifted + cell_code[i];
                            octree.nodes[offset + i] = 0;
                            octree.parent[offset + i] = task;
                            mask |= (1 << cell_code[i]);
                        }
                        octree.nodes[task] = (offset << 8) + mask;
                    }
                    else
                        octree.nodes[task] = 0;
                }

                __syncthreads();
                if (threadIdx.x == CTA_SIZE - 1)
                {
                    total_new += cell_count + offsets[threadIdx.x];
                    nodes_num += cell_count + offsets[threadIdx.x];
                }
                __syncthreads();

            }

            if (threadIdx.x == CTA_SIZE - 1)
            {
                tasks_beg = tasks_end;
                tasks_end += total_new;
                total_new = 0;
            }
            ++level;
            __syncthreads();
        }

        if (threadIdx.x == CTA_SIZE - 1)
            *octree.nodes_num = nodes_num;
    }
};

void Octree::setCloud(const float4* Points, const int poinm) {
    points.upload(Points, poinm);
    cudaDeviceSynchronize();
}

__global__ void  singleStepKernel(const SingleStepBuild ssb) { ssb.build(); }
void Octree::build() {
    Clock time;
    time.begin();
    int points_num = (int)points.size();
    const int transaction_size = 128 / sizeof(int);
    int cols = std::max<int>(points_num, transaction_size * 4);
    int rows = 10 + 1; // = 13
    storage.create(rows, cols);
    codes = DeviceArray<int>(storage.ptr(0), points_num);
    indices = DeviceArray<int>(storage.ptr(1), points_num);
    octree.nodes = storage.ptr(2);
    octree.codes = storage.ptr(3);
    octree.begs = storage.ptr(4);
    octree.ends = storage.ptr(5);
    octree.parent = storage.ptr(6);
    octree.nodes_num = storage.ptr(7);
    points_sorted = DeviceArray2D<float>(3, points_num, storage.ptr(8), storage.step());

    //找到该三维点云中的最大和最小x,y,z的值
    thrust::device_ptr<PointType> beg(points.ptr());
    thrust::device_ptr<PointType> end = beg + points.size();


    PointType atmax, atmin;
    atmax.x = atmax.y = atmax.z = std::numeric_limits<float>::max();
    atmin.x = atmin.y = atmin.z = std::numeric_limits<float>::lowest();
    atmax.w = atmin.w = 0;

    //ScopeTimer timer("reduce"); 
    PointType minp = thrust::reduce(beg, end, atmax, SelectMinPoint<PointType>());
    PointType maxp = thrust::reduce(beg, end, atmin, SelectMaxPoint<PointType>());
    octree.minp = make_float3(minp.x, minp.y, minp.z);
    octree.maxp = make_float3(maxp.x, maxp.y, maxp.z);


    thrust::device_ptr<int> codes_beg(codes.ptr());
    thrust::device_ptr<int> codes_end = codes_beg + codes.size();
    //ScopeTimer timer("morton"); 
    thrust::transform(beg, end, codes_beg, CalcMorton(octree.minp, octree.maxp));

    thrust::device_ptr<int> indices_beg(indices.ptr());
    thrust::device_ptr<int> indices_end = indices_beg + indices.size();

    //ScopeTimer timer("sort"); 
    //0,1,2,3,4,5,……
    thrust::sequence(indices_beg, indices_end);
    // indices_beg[i]保存的是当前值在排序前的索引 排序前code[indices[i]]=排序后code[i]
    thrust::sort_by_key(codes_beg, codes_end, indices_beg);


    thrust::device_ptr<float> xs(points_sorted.ptr(0));
    thrust::device_ptr<float> ys(points_sorted.ptr(1));
    thrust::device_ptr<float> zs(points_sorted.ptr(2));

    //ScopeTimer timer("perm2"); 
    //beg没有排序，所以beg对应的是之前的code顺序，这样做使得beg按照现在code找到之前的code进行顺序操作
    thrust::transform(make_permutation_iterator(beg, indices_beg),
        make_permutation_iterator(end, indices_end),
        make_zip_iterator(thrust::make_tuple(xs, ys, zs)), PointType_to_tuple<PointType>());

    SingleStepBuild ssb;
    ssb.octree = octree;
    ssb.codes = codes;
    ssb.points_number = (int)codes.size();
    //printFuncAttrib(singleStepKernel);
    time.end();
    time.displayInterval("处理建树数据花费");
    time.begin();
    singleStepKernel << <GRID_SIZE, CTA_SIZE >> > (ssb);
    cudaGetLastError();
    cudaDeviceSynchronize();
    time.end();
    time.displayInterval("建树kernel花费");
}

__global__ void KernelKNN(const KBatch batch)
{
    const int query_index = blockIdx.x * blockDim.x + threadIdx.x;

    const bool active = query_index < batch.queries_num;

    if (__all_sync(0xFFFFFFFF, active == false))
        return;

    Warp_knnSearch search(batch, query_index);
    search.launch(active);
}
void Octree::Knn_Search(const float4* Points, const int query_num, const int k, float3* rsnormals) {
    Queries queries;
    queries.upload(Points, query_num);
    DeviceArray<float>knndist;
    DeviceArray<int>knnindices;
    NorMals normals;
    normals.create(queries.size());
    knndist.create(queries.size() * k);
    knnindices.create(queries.size() * k);
    KBatch batch;
    batch.octree = octree;
    batch.queries_num = (int)queries.size();
    batch.queries = queries;
    batch.normals = normals;
    batch.points = points_sorted;
    batch.k = k;
    batch.points_step = points_sorted.step() / points_sorted.elem_size;
    batch.knndist = knndist;
    batch.knnindices = knnindices;
    int block = 32;
    int grid = (batch.queries_num + block - 1) / block;
    KernelKNN << < grid, 32 >> > (batch);
    normals.download(rsnormals);
    cudaGetLastError();
    cudaDeviceSynchronize();
}

__global__ void KernelRS(const Batch batch)
{
    const int query_index = blockIdx.x * blockDim.x + threadIdx.x;

    const bool active = query_index < batch.queries_num;

    if (__all_sync(0xFFFFFFFF, active == false))
        return;

    Warp_radiusSearch search(batch, query_index);
    search.launch(active);
}
void Octree::radiusSearch(float4* Points, const int query_num, const float radius, const int max, float3* rsnormals)
{
    Clock time;
    Queries queries;
    queries.upload(Points, query_num);
    NorMals normals;
    normals.create(queries.size());
    DeviceArray<float4>newquerys;
    newquerys.create(queries.size());
    Batch batch;
    batch.octree = octree;
    batch.queries_num = query_num;
    batch.queries = queries;
    batch.normals = normals;
    batch.points = points_sorted;
    batch.max_results = max;
    batch.radiuses = radius;
    cudaFuncSetCacheConfig(KernelRS, cudaFuncCachePreferL1);
    int block = KernelPolicy::CTA_SIZE;
    int grid = divUp(query_num, block);
    time.begin();
    KernelRS << <grid, block >> > (batch);
    cudaDeviceSynchronize();
    time.end();
    time.displayInterval("法线kernel花费");
    normals.download(rsnormals);
    queries.download(Points);
    cudaGetLastError();
    cudaDeviceSynchronize();
}


