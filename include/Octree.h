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
struct OctreeGlobal//ȫ�ְ˲���
{
    int* nodes;         //�˲����ĸ����ڵ�  ǰ8λ��ʾ�ڰ˲����Ľڵ��ţ����λ��ʾ�Ӻ���

    int* codes;         //�ڵ���� ÿ3λ����һ���㼶����10��

    int* begs;          //ÿ���ڵ��Ӧ�ĵ��ƿ�ʼ����

    int* ends;          //ÿ���ڵ��Ӧ�ĵ��ƽ�������

    int* nodes_num;		//�ڵ����

    int* parent;        //ÿ���ڵ�ĸ��ڵ�

    float3 minp, maxp;	//ÿ���ڵ�����귶Χ

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


