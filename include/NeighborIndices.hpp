#pragma once
#include"Device_Array.hpp"
struct NeighborIndices
{
    DeviceArray<int> data;
    DeviceArray<int> sizes;
    int max_elems;

    NeighborIndices() {}
    NeighborIndices(int query_number, int max_elements) : max_elems(0)
    {
        create(query_number, max_elements);
    }

    void create(int query_number, int max_elements)
    {
        max_elems = max_elements;
        data.create(query_number * max_elems);

        if (max_elems != 1)
            sizes.create(query_number);
    }

    void upload(const std::vector<int>& data, const std::vector<int>& sizes, int max_elements)
    {
        this->data.upload(data);
        this->sizes.upload(sizes);
        max_elems = max_elements;
    }

    bool validate(std::size_t cloud_size) const
    {
        return (sizes.size() == cloud_size) && (cloud_size * max_elems == data.size());
    }

    operator PtrStep<int>() const
    {
        return { (int*)data.ptr(), max_elems * sizeof(int) };
    }

    std::size_t neighboors_size() const { return data.size() / max_elems; }
};