#pragma once
#include<iostream>
#include <atomic>
#include <thread>
#include "cuda_runtime.h"
#include"device_pointer.hpp"
class  DeviceMemory {
public:
    __host__ __device__ DeviceMemory() : data_(nullptr), sizeBytes_(0), refcount_(nullptr) {};

    __host__ __device__ ~DeviceMemory() { release(); };

    __host__ __device__ DeviceMemory(std::size_t sizeBytes_arg) : data_(nullptr), sizeBytes_(0), refcount_(nullptr) { create(sizeBytes_arg); };

    __host__ __device__ DeviceMemory(void* ptr_arg, std::size_t sizeBytes_arg) : data_(ptr_arg), sizeBytes_(sizeBytes_arg), refcount_(nullptr) {};

    __host__ __device__ DeviceMemory(const DeviceMemory& other_arg) : data_(other_arg.data_)
        , sizeBytes_(other_arg.sizeBytes_)
        , refcount_(other_arg.refcount_) {
        if (refcount_)
            refcount_->fetch_add(1);
    };

    __host__ __device__ DeviceMemory& operator=(const DeviceMemory& other_arg) {
        if (this != &other_arg) {
            if (other_arg.refcount_)
                other_arg.refcount_->fetch_add(1);
            release();

            data_ = other_arg.data_;
            sizeBytes_ = other_arg.sizeBytes_;
            refcount_ = other_arg.refcount_;
        }
        return *this;
    };

    __host__ __device__ void create(std::size_t sizeBytes_arg)
    {
        if (sizeBytes_arg == sizeBytes_)
            return;

        if (sizeBytes_arg > 0) {
            if (data_)
                release();

            sizeBytes_ = sizeBytes_arg;

            cudaMalloc(&data_, sizeBytes_);

            // refcount_ = (int*)cv::fastMalloc(sizeof(*refcount_));
            refcount_ = new std::atomic<int>(1);
        }
    };

    __host__ __device__ void release() {
        if (refcount_ && refcount_->fetch_sub(1) == 1) {
            // cv::fastFree(refcount);
            delete refcount_;
            cudaFree(data_);
        }
        data_ = nullptr;
        sizeBytes_ = 0;
        refcount_ = nullptr;
    };

    void copyTo(DeviceMemory& other, std::size_t sizeBytes) const;

    void upload(const void* host_ptr_arg, std::size_t sizeBytes_arg);

    bool upload(const void* host_ptr,
            std::size_t device_begin_byte_offset,
            std::size_t num_bytes);

    void download(void* host_ptr_arg) const;

    bool download(void* host_ptr,
            std::size_t device_begin_byte_offset,
            std::size_t num_bytes) const;

    void swap(DeviceMemory& other_arg);

    
    template <class T>
    __host__ __device__ T* ptr() { return (T*)data_; };

    template <class T>
    __host__ __device__ const T* ptr() const { return (const T*)data_; };

    template <class U>
    __host__ __device__ operator PtrSz<U>() const {
        PtrSz<U> result;
        result.data = (U*)ptr<U>();
        result.size = sizeBytes_ / sizeof(U);
        return result;
    };

    __host__ __device__ bool empty() const { return !data_; };

    __host__ __device__ std::size_t sizeBytes() const { return sizeBytes_; };

private:

    void* data_;//void * data 是一个无类型的指针参数，任意类型指针可以赋值给data

    std::size_t sizeBytes_;

    std::atomic<int>* refcount_;
};


//二维内存
class  DeviceMemory2D {
public:
    __host__ __device__  DeviceMemory2D() : data_(nullptr), step_(0), colsBytes_(0), rows_(0), refcount_(nullptr) {};

    __host__ __device__  ~DeviceMemory2D(){ release(); };

    __host__ __device__  DeviceMemory2D(int rows_arg, int colsBytes_arg) : data_(nullptr), step_(0), colsBytes_(0), rows_(0), refcount_(nullptr)
    {
        create(rows_arg, colsBytes_arg);
    };

    __host__ __device__  DeviceMemory2D(int rows_arg, int colsBytes_arg, void* data_arg, std::size_t step_arg) : data_(data_arg)
        , step_(step_arg)
        , colsBytes_(colsBytes_arg)
        , rows_(rows_arg)
        , refcount_(nullptr)
    {};

    __host__ __device__  DeviceMemory2D(const DeviceMemory2D& other_arg) : data_(other_arg.data_)
        , step_(other_arg.step_)
        , colsBytes_(other_arg.colsBytes_)
        , rows_(other_arg.rows_)
        , refcount_(other_arg.refcount_)
    {
        if (refcount_)
            refcount_->fetch_add(1);
    };

    __host__ __device__  DeviceMemory2D& operator=(const DeviceMemory2D& other_arg)
        {
            if (this != &other_arg) {
                if (other_arg.refcount_)
                    other_arg.refcount_->fetch_add(1);
                release();

                colsBytes_ = other_arg.colsBytes_;
                rows_ = other_arg.rows_;
                data_ = other_arg.data_;
                step_ = other_arg.step_;

                refcount_ = other_arg.refcount_;
            }
            return *this;
        };

    __host__ __device__  void create(int rows_arg, int colsBytes_arg)
        {
            if (colsBytes_ == colsBytes_arg && rows_ == rows_arg)
                return;

            if (rows_arg > 0 && colsBytes_arg > 0) {
                if (data_)
                    release();

                colsBytes_ = colsBytes_arg;
                rows_ = rows_arg;

                cudaMallocPitch((void**)&data_, &step_, colsBytes_, rows_);

                refcount_ = new std::atomic<int>(1);
            }
        }
        ;

    __host__ __device__  void release()
    {
        if (refcount_ && refcount_->fetch_sub(1) == 1) {
            delete refcount_;
           cudaFree(data_);
        }

        colsBytes_ = 0;
        rows_ = 0;
        data_ = nullptr;
        step_ = 0;
        refcount_ = nullptr;
    };

    void copyTo(DeviceMemory2D& other) const;

    void upload(const void* host_ptr_arg,
            std::size_t host_step_arg,
            int rows_arg,
            int colsBytes_arg);

    void download(void* host_ptr_arg, std::size_t host_step_arg) const;

    void swap(DeviceMemory2D& other_arg);

    template <class T>
    __host__ __device__  T* ptr(int y_arg = 0) {
        return (T*)((char*)data_ + y_arg * step_);
    };

    template <class T>
    __host__ __device__  const T* ptr(int y_arg = 0) const {
        return (const T*)((const char*)data_ + y_arg * step_);
    };

    template <class U>
    __host__ __device__  operator PtrStep<U>() const {
        PtrStep<U> result;
        result.data = (U*)ptr<U>();
        result.step = step_;
        return result;
    };

    template <class U>
    __host__ __device__  operator PtrStepSz<U>() const {
        PtrStepSz<U> result;
        result.data = (U*)ptr<U>();
        result.step = step_;
        result.cols = colsBytes_ / sizeof(U);
        result.rows = rows_;
        return result;
    };

    __host__ __device__ bool empty() const {
        return !data_;
    };

    __host__ __device__ int colsBytes() const {
        return colsBytes_;
    };

    __host__ __device__ int rows() const {
        return rows_;
    };

    __host__ __device__ std::size_t step() const {
        return step_;
    };

private:
    void* data_;

    std::size_t step_;

    int colsBytes_;

    int rows_;

    std::atomic<int>* refcount_;
};



