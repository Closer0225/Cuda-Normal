#pragma once
//设备指针
template <typename T>
struct DevPtr {
    using elem_type = T;
    const static std::size_t elem_size = sizeof(elem_type);

    T* data;
    __host__ __device__ __forceinline__
    DevPtr() : data(nullptr) {}

    __host__ __device__ __forceinline__
    DevPtr(T* data_arg) : data(data_arg) {}

    __host__ __device__ __forceinline__
    std::size_t elemSize() const
    {
        return elem_size;
    }

    __host__ __device__ __forceinline__
    operator T* () { return data; }

    __host__ __device__ __forceinline__
    operator const T* () const { return data; }
};

//带有大小的设备指针
template <typename T>
struct PtrSz : public DevPtr<T> {
    __host__ __device__ __forceinline__
    PtrSz() : size(0) {}

    __host__ __device__ __forceinline__ 
    PtrSz(T* data_arg, std::size_t size_arg) : DevPtr<T>(data_arg), size(size_arg) {}

    std::size_t size;
};

//带有字节的单位指针
template <typename T>
struct PtrStep : public DevPtr<T> {
    __host__ __device__ __forceinline__
    PtrStep() : step(0) {}

    __host__ __device__ __forceinline__
    PtrStep(T* data_arg, std::size_t step_arg) : DevPtr<T>(data_arg), step(step_arg) {}

    //以字节为单位的两个连续行之间的短跨步。始终存储步长，并且以字节为单位！
    std::size_t step;

    __host__ __device__ __forceinline__ T* ptr(int y = 0)
    {
        return (T*)((char*)DevPtr<T>::data + y * step);
    }

    __host__ __device__ __forceinline__ const T* ptr(int y = 0) const
    {
        return (const T*)((const char*)DevPtr<T>::data + y * step);
    }
    __host__ __device__ __forceinline__ T& operator()(int y, int x)
    {
        return ptr(y)[x];
    }
    __host__ __device__ __forceinline__ const T& operator()(int y, int x) const
    {
        return ptr(y)[x];
    }
};
template <typename T>
struct PtrStepSz : public PtrStep<T> {
    __host__ __device__ __forceinline__
    PtrStepSz() : cols(0), rows(0) {}

    __host__ __device__ __forceinline__
    PtrStepSz(int rows_arg, int cols_arg, T* data_arg, std::size_t step_arg)
        : PtrStep<T>(data_arg, step_arg), cols(cols_arg), rows(rows_arg)
    {}

    int cols;
    int rows;
};
