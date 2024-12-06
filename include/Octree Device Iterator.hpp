#pragma once
#include"Octree.h"
//设备端八叉树遍历迭代器
struct OctreeIteratorDeviceNS
{
	int level;
	int node_idx;
	int length;
	const OctreeGlobal& octree;

	__device__ __forceinline__ OctreeIteratorDeviceNS(const OctreeGlobal& octree_arg) : octree(octree_arg)
	{
		node_idx = 0;
		level = 0;
		length = 1;
	}

	__device__ __forceinline__ void gotoNextLevel(int first, int len)
	{
		node_idx = first;
		length = len;
		++level;
	}

	__device__ __forceinline__ int operator*() const
	{
		return node_idx;
	}

	__device__ __forceinline__ void operator++()
	{
#if 1
		while (level >= 0)
		{
			if (length > 1)
			{
				length--;
				node_idx++;
				break;
			}

			if (node_idx == 0)
			{
				level = -1;
				return;
			}
			//跳到父节点
			node_idx = octree.parent[node_idx];
			--level;
			//判断父节点是否为根节点
			if (node_idx == 0)
			{
				level = -1;
				return;
			}
			int parent = octree.nodes[octree.parent[node_idx]];
			int parent_first = parent >> 8;
			int parent_len = __popc(parent & 0xFF);
			int pos = node_idx - parent_first;
			length = parent_len - pos;
		}
#endif
	}

};
