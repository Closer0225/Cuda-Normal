#define _CRT_SECURE_NO_WARNINGS
#include"Octree.h"
#include"Clock.h"
#include<iostream>
using namespace std;
void openPointCloud(const std::string& fileName, pointcloud& pPoints, pointcloud& nNormals);
string savePointCloud(const std::string& fileName, const float4* pPoints, const float3* rsnormals,int query_num);
int main() {
	pointcloud host_points;
	pointcloud host_normals;
	Clock time;
	time.begin();
	openPointCloud("../data/happy_vrip.asc", host_points, host_normals);
	time.end();
	time.displayInterval("加载点云文件");
	int poinm = host_points.size();
	float4* Points = new float4[poinm];
	for (int i = 0; i < poinm; i++) {
		Points[i] = make_float4((host_points)[i].x, (host_points)[i].y, (host_points)[i].z, 0);
	}
	float radius = 1.0;
	Octree octree;
	time.begin();
	octree.setCloud(Points, poinm);
	time.end();
	time.displayInterval("传输数据共花费");
	time.begin();
	octree.build();
	time.end();
	time.displayInterval("建树共花费");
	time.begin();
	float3* rsnormals = new float3[poinm];
	octree.radiusSearch(Points, poinm, radius, maxresult, rsnormals);
	time.end();
	time.displayInterval("计算法线共花费");
	savePointCloud("../data/Pointnormals.asc", Points,rsnormals,poinm);
	delete[]Points;
	delete[]rsnormals;
	return 0;
}

void openPointCloud(const std::string& fileName, pointcloud& pPoints, pointcloud& nNormals)
{
	int i = 0;
	std::string errorInfo;
	FILE* fp = fopen(fileName.c_str(), "r");
	if (fp)
	{
		try
		{
			float3 point;
			float3 normal;
			char line[1024];

			while (fgets(line, 1023, fp))
			{
				if (sscanf(line, "%f%f%f%f%f%f", &point.x, &point.y, &point.z, &normal.x, &normal.y, &normal.z) == 6)
				{
					pPoints.push_back(point);
					nNormals.push_back(normal);
				}
				else if (sscanf(line, "%f%f%f", &point.x, &point.y, &point.z) == 3)
					pPoints.push_back(make_float3(point.x, point.y, point.z));
				i++;
			}
		}
		catch (std::exception& error)
		{
			errorInfo = error.what();
		}
		fclose(fp);
	}
	else
		errorInfo = "打开点云文件失败！\n";
}

std::string savePointCloud(const std::string& fileName, const float4* pPoints,const float3*rsnormals,int query_num) {
	//需要从utf8转成ascii
	FILE* fp = fopen(fileName.c_str(), "w");
	if (fp)
	{

		for (auto i = 0; i < query_num; ++i)
		{
			fprintf(fp, "%f %f %f %f %f %f\n", pPoints[i].x, pPoints[i].y,
				pPoints[i].z,rsnormals[i].x,rsnormals[i].y,rsnormals[i].z);
		}
		fclose(fp);
	}
	else
		return "保存点云文件失败！\n";
	return "";
}

