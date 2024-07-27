#include"Device_Memory.h"
void DeviceMemory::copyTo(DeviceMemory& other, std::size_t sizeBytes) const
{
    if (empty())
        other.release();
    else {
        other.create(sizeBytes);
        cudaMemcpy(other.data_, data_, sizeBytes, cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }
}       

void DeviceMemory::upload(const void* host_ptr_arg, std::size_t sizeBytes_arg)
{
    create(sizeBytes_arg);
    cudaMemcpy(data_, host_ptr_arg, sizeBytes_, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

bool DeviceMemory::upload(const void* host_ptr_arg,
    std::size_t device_begin_byte_offset,
    std::size_t num_bytes)
{
    if (device_begin_byte_offset + num_bytes > sizeBytes_) {
        return false;
    }
    void* begin = static_cast<char*>(data_) + device_begin_byte_offset;
    cudaMemcpy(begin, host_ptr_arg, num_bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    return true;
}

void DeviceMemory::download(void* host_ptr_arg) const
{
    cudaMemcpy(host_ptr_arg, data_, sizeBytes_, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

bool DeviceMemory::download(void* host_ptr_arg,
    std::size_t device_begin_byte_offset,
    std::size_t num_bytes) const
{
    if (device_begin_byte_offset + num_bytes > sizeBytes_) {
        return false;
    }
    const void* begin = static_cast<char*>(data_) + device_begin_byte_offset;
    cudaMemcpy(host_ptr_arg, begin, num_bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return true;
}

void DeviceMemory::swap(DeviceMemory& other_arg)
{
    std::swap(data_, other_arg.data_);
    std::swap(sizeBytes_, other_arg.sizeBytes_);
    std::swap(refcount_, other_arg.refcount_);
}
void DeviceMemory2D::copyTo(DeviceMemory2D& other) const
{
    if (empty())
        other.release();
    else {
        other.create(rows_, colsBytes_);
        cudaMemcpy2D(other.data_,
            other.step_,
            data_,
            step_,
            colsBytes_,
            rows_,
            cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
    }
}

void DeviceMemory2D::upload(const void* host_ptr_arg,
    std::size_t host_step_arg,
    int rows_arg,
    int colsBytes_arg)
{
    create(rows_arg, colsBytes_arg);
    cudaMemcpy2D(data_,
        step_,
        host_ptr_arg,
        host_step_arg,
        colsBytes_,
        rows_,
        cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

void DeviceMemory2D::download(void* host_ptr_arg, std::size_t host_step_arg) const
{
    cudaMemcpy2D(host_ptr_arg,
        host_step_arg,
        data_,
        step_,
        colsBytes_,
        rows_,
        cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

void DeviceMemory2D::swap(DeviceMemory2D& other_arg)
{
    std::swap(data_, other_arg.data_);
    std::swap(step_, other_arg.step_);

    std::swap(colsBytes_, other_arg.colsBytes_);
    std::swap(rows_, other_arg.rows_);
    std::swap(refcount_, other_arg.refcount_);
}
