#pragma once
#include <limits>
#include"NeighborIndices.hpp"
#include<string>
using pointcloud = std::vector<float3>;
using NorMals = DeviceArray<float3>;
using PointType = float4;
using PointCloud = DeviceArray<float4>;
using Queries = DeviceArray<float4>;
using Radiuses = DeviceArray<float>;
using BatchResult = DeviceArray<int>;
using BatchResultSizes = DeviceArray<int>;
using BatchResultSqrDists = DeviceArray<float>;
using Indices = DeviceArray<int>;
using ResultSqrDists = DeviceArray<float>;
enum MyEnum
{
    IN_MAX =(std::numeric_limits<int>::max)(),
    maxresult =150,
    order = 2,
};
struct OctreeGlobal//全局八叉树
{
    int* nodes;         //八叉树的各个节点  前8位表示在八叉树的节点编号，后八位表示子孩子

    int* codes;         //节点编码 每3位代表一个层级，共10层

    int* begs;          //每个节点对应的点云开始索引

    int* ends;          //每个节点对应的点云结束索引

    int* nodes_num;		//节点个数

    int* parent;        //每个节点的父节点

    float3 minp, maxp;	//每个节点的坐标范围

    OctreeGlobal() : nodes(nullptr), begs(nullptr), codes(nullptr), ends(nullptr), nodes_num(nullptr), parent(nullptr), minp(make_float3(0, 0, 0)), maxp(make_float3(0, 0, 0)) {}
};
class  Octree
{
public:

    Octree() {};

    virtual ~Octree() { };

    void setCloud(const float4* Points,const int poinm) ;

    void build();

    void radiusSearch(float4* Points, const int query_num, const float radius, const int max, float3* rsnormals);

    void Knn_Search(const float4* Points,const int query_num,const int k, float3* rsnormals);

    PointCloud points;

    OctreeGlobal octree;

    DeviceArray2D<int> storage;

    DeviceArray2D<float> points_sorted;

    DeviceArray<int> codes;

    DeviceArray<int> indices;
};


