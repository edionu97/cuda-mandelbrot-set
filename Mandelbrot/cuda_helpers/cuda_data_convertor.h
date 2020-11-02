#pragma once
#include <cuda_runtime_api.h>
#include <exception>
#include <vector>

#include "cuda_device_helpers.h"

template<typename T> class cuda_data_convertor {
public:

	/// <summary>
	/// Define the device_data_wrapper
	/// </summary>
	class device_data_wrapper {
		T* device_data_ = nullptr;

	public:
		/// <summary>
		/// Copy not allowed
		/// </summary>
		/// <param name="other">representing the other data</param>
		device_data_wrapper(const device_data_wrapper& other) = delete;
		device_data_wrapper& operator=(const device_data_wrapper& other) = delete;
		device_data_wrapper& operator=(device_data_wrapper&& other) = delete;

		device_data_wrapper() = default;

		device_data_wrapper(T* device_data) noexcept
		{
			device_data_ = device_data;
		}

		device_data_wrapper(device_data_wrapper&& other) noexcept
		{
			device_data_ = other.device_data_;
		}

		T* get_device_data() const
		{
			return device_data_;
		}

		~device_data_wrapper()
		{
			if (device_data_ == nullptr)
			{
				return;
			}

			cudaFree(device_data_);
			device_data_ = nullptr;
		}
	};

	/// <summary>
	/// Converts an stl vector to device array pinter
	/// </summary>
	/// <param name="array">the vector that will be converted</param>
	/// <returns>the device data</returns>
	static device_data_wrapper convert_stl_vector_to_device_array(const std::vector<T>& array) noexcept
	{
		//device pointer
		T* device_array_pointer = nullptr;

		try
		{
			//define the number of bytes that will be copied
			const auto bytes_size = sizeof(T) * array.size();
			//allocate memory
			cuda_device_helpers::check(cudaMalloc(reinterpret_cast<void**>(&device_array_pointer), bytes_size));
			//copy values
			cuda_device_helpers::check(cudaMemcpy(device_array_pointer, &array[0], bytes_size, cudaMemcpyHostToDevice));
		}
		catch (std::exception& e)
		{
			//if the pointer is not allocated than return device_data of null_ptr
			if (device_array_pointer == nullptr)
			{
				throw e;
			}

			cudaFree(device_array_pointer);
			throw e;
		}

		//create a device_data_wrapper
		return { device_array_pointer };
	}

	/// <summary>
	/// Convert device data to stl vector
	/// </summary>
	/// <param name="device_data_array">the device data</param>
	/// <param name="array_size">the size of the array</param>
	/// <returns></returns>
	static std::vector<T> convert_device_data_to_stl_vector(const device_data_wrapper& device_data_array, const size_t array_size)
	{
		//create the host array data, and allocate it
		auto* host_array_data = new T[array_size];

		try
		{
			//copy data back from device to host
			const int byte_size = array_size * sizeof(T);
			const auto* source = device_data_array.get_device_data();
			cuda_device_helpers::check(cudaMemcpy(host_array_data, source, byte_size, cudaMemcpyDeviceToHost));

			//create the vector
			std::vector<T> data(host_array_data, host_array_data + array_size);

			//deallocate the allocated memory
			delete[] host_array_data;

			//teturn data
			return  data;
		}
		catch (std::exception& e)
		{
			//deallocate the allocated memory
			delete[] host_array_data;

			//throw the exception back
			throw e;
		}
	}
};
