#pragma once
#include"Device_Memory.h"
#include <vector>
template <class T>
class  DeviceArray : public DeviceMemory {
public:
    using type = T;

    enum { elem_size = sizeof(T) };

    __host__ __device__ DeviceArray() {};

    __host__ __device__ DeviceArray(std::size_t size) : DeviceMemory(size* elem_size) {};

    __host__ __device__ DeviceArray(T* ptr, std::size_t size) : DeviceMemory(ptr, size* elem_size) {};

    __host__ __device__ DeviceArray(const DeviceArray& other) : DeviceMemory(other) {};

    __host__ __device__ DeviceArray& operator=(const DeviceArray& other) {
        DeviceMemory::operator=(other);
        return *this;
    };

    void create(std::size_t size);

    void release();

    void copyTo(DeviceArray& other, std::size_t sizeBytes_) const;

    void upload(const T* host_ptr, std::size_t size);

    bool upload(const T* host_ptr, std::size_t device_begin_offset, std::size_t num_elements);

    void download(T* host_ptr) const;

    bool download(T* host_ptr,
            std::size_t device_begin_offset,
            std::size_t num_elements) const;

    template <class A>
    void upload(const std::vector<T, A>& data);

    template <typename A>
    void download(std::vector<T, A>& data) const;

    void swap(DeviceArray& other_arg);

    __host__ __device__
        T* ptr() { return DeviceMemory::ptr<T>(); };

    __host__ __device__
        const T* ptr() const { return DeviceMemory::ptr<T>(); };

    __host__ __device__
        operator T* () { return ptr(); };

    __host__ __device__
        operator const T* () const { return ptr(); };
    
    __host__ __device__
        std::size_t size() const { return sizeBytes() / elem_size; };
};


template <class T>
inline void
DeviceArray<T>::create(std::size_t size)
{
    DeviceMemory::create(size * elem_size);
}

template <class T>
inline void
DeviceArray<T>::release()
{
    DeviceMemory::release();
}

template <class T>
inline void
DeviceArray<T>::copyTo(DeviceArray& other, std::size_t sizeBytes_) const
{
    DeviceMemory::copyTo(other,sizeBytes_);
}

template <class T>
inline void
DeviceArray<T>::upload(const T* host_ptr, std::size_t size)
{
    DeviceMemory::upload(host_ptr, size * elem_size);
}

template <class T>
inline bool
DeviceArray<T>::upload(const T* host_ptr,
    std::size_t device_begin_offset,
    std::size_t num_elements)
{
    std::size_t begin_byte_offset = device_begin_offset * sizeof(T);
    std::size_t num_bytes = num_elements * sizeof(T);
    return DeviceMemory::upload(host_ptr, begin_byte_offset, num_bytes);
}

template <class T>
inline void
DeviceArray<T>::download(T* host_ptr) const
{
    DeviceMemory::download(host_ptr);
}

template <class T>
inline bool
DeviceArray<T>::download(T* host_ptr,
    std::size_t device_begin_offset,
    std::size_t num_elements) const
{
    std::size_t begin_byte_offset = device_begin_offset * sizeof(T);
    std::size_t num_bytes = num_elements * sizeof(T);
    return DeviceMemory::download(host_ptr, begin_byte_offset, num_bytes);
}

template <class T>
void
DeviceArray<T>::swap(DeviceArray& other_arg)
{
    DeviceMemory::swap(other_arg);
}

template <class T>
template <class A>
inline void
DeviceArray<T>::upload(const std::vector<T, A>& data)
{
    upload(&data[0], data.size());
}

template <class T>
template <class A>
inline void
DeviceArray<T>::download(std::vector<T, A>& data) const
{
    data.resize(size());
    if (!data.empty())
        download(&data[0]);
}


//¶þÎ¬Êý×é
template <class T>
class  DeviceArray2D : public DeviceMemory2D {
public:
    using type = T;

    enum { elem_size = sizeof(T) };

    __host__ __device__ DeviceArray2D() {};

    __host__ __device__ DeviceArray2D(int rows, int cols) 
        : DeviceMemory2D(rows, cols* elem_size)
    {};

    __host__ __device__ DeviceArray2D(int rows, int cols, void* data, std::size_t stepBytes) 
        : DeviceMemory2D(rows, cols* elem_size, data, stepBytes)
    {};

    __host__ __device__ DeviceArray2D(const DeviceArray2D& other)
        : DeviceMemory2D(other)
    {};

    __host__ __device__ DeviceArray2D& operator=(const DeviceArray2D& other)
    {
        DeviceMemory2D::operator=(other);
        return *this;
    };

     void create(int rows, int cols);

     void release();

     void copyTo(DeviceArray2D& other) const;

     void upload(const void* host_ptr, std::size_t host_step, int rows, int cols);

     void download(void* host_ptr, std::size_t host_step) const;

     void swap(DeviceArray2D& other_arg);

     template <class A>
     void upload(const std::vector<T, A>& data, int cols);

     template <class A>
     void download(std::vector<T, A>& data, int& cols) const;

    __host__ __device__ T* ptr(int y=0)
    {
        return DeviceMemory2D::ptr<T>(y);
    };

    __host__ __device__ const T* ptr(int y =0) const
    {
        return DeviceMemory2D::ptr<T>(y);   
    };

    __host__ __device__ operator T* ()
    {
        return ptr();
    };

    __host__ __device__  operator const T* () const
    {
        return ptr();
    };

    __host__ __device__  int cols() const
    {
        return DeviceMemory2D::colsBytes() / elem_size;
    };

    __host__ __device__ int rows() const
    {
        return DeviceMemory2D::rows();
    };

    __host__ __device__ std::size_t elem_step() const
    {
        return DeviceMemory2D::step() / elem_size;
    };
};
template <class T>
inline void
DeviceArray2D<T>::create(int rows, int cols)
{
    DeviceMemory2D::create(rows, cols * elem_size);
}
template <class T>
inline void
DeviceArray2D<T>::release()
{
    DeviceMemory2D::release();
}
template <class T>
inline void
DeviceArray2D<T>::copyTo(DeviceArray2D& other) const
{
    DeviceMemory2D::copyTo(other);
}
template <class T>
inline void
DeviceArray2D<T>::upload(const void* host_ptr,
    std::size_t host_step,
    int rows,
    int cols)
{
    DeviceMemory2D::upload(host_ptr, host_step, rows, cols * elem_size);
}
template <class T>
inline void
DeviceArray2D<T>::download(void* host_ptr, std::size_t host_step) const
{
    DeviceMemory2D::download(host_ptr, host_step);
}
template <class T>
template <class A>
inline void
DeviceArray2D<T>::upload(const std::vector<T, A>& data, int cols)
{
    upload(&data[0], cols * elem_size, data.size() / cols, cols);
}
template <class T>
template <class A>
inline void
DeviceArray2D<T>::download(std::vector<T, A>& data, int& elem_step) const
{
    elem_step = cols();
    data.resize(cols() * rows());
    if (!data.empty())
        download(&data[0], colsBytes());
}

template <class T>
void
DeviceArray2D<T>::swap(DeviceArray2D& other_arg)
{
    DeviceMemory2D::swap(other_arg);
}