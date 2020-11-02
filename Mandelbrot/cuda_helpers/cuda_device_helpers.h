#pragma once
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <exception>
#include <map>
#include <sstream>

typedef std::pair<cudaDeviceProp, int> device_info;


class cuda_device_helpers
{
public:

	class device_interactions
	{
		std::map<std::string, device_info> installed_devices_;
	public:

		/// <summary>
		/// This method it is used for adding a specific device into the device_list
		/// </summary>
		/// <param name="device_name">the name of the device</param>
		/// <param name="device_properties">device's properties</param>
		void add_device(const std::string& device_name, const device_info& device_properties) noexcept
		{
			//add only not existing devices
			const auto it = installed_devices_.find(device_name);
			if(it != installed_devices_.end())
			{
				return;
			}

			//insert into map the association between device name and device properties
			installed_devices_.insert(std::make_pair(device_name, device_properties));
		}

		/// <summary>
		/// sets the device
		/// </summary>
		/// <param name="device_name"></param>
		void set_device(const std::string& device_name) const
		{
			//get device info
			const auto& device_info = get_device_info(device_name);

			//get device no
			const auto& device_no = device_info.second;

			//set the device
			check(cudaSetDevice(device_no));
		}

		//resets the device
		void reset_device(const std::string& device_name) const
		{
			//get device info
			const auto& device_info = get_device_info(device_name);

			//get device no
			const auto& device_no = device_info.second;

			//set the device
			check(cudaDeviceReset());
		}

		/// <summary>
		/// Get information about device
		/// </summary>
		/// <param name="device_name">the name of the device</param>
		/// <returns>device properties or throws an exception if a device with that name does not exist</returns>
		const device_info& get_device_info(const std::string& device_name) const
		{
			//get the pointer to the element from map
			const auto it = installed_devices_.find(device_name);

			if (it != installed_devices_.end())
			{
				return it->second;
			}

			throw std::exception("Item not found");
		}

		/// <summary>
		/// Get properties for a specific device
		/// </summary>
		/// <param name="device_name">the name of the device</param>
		/// <returns>device properties or throws an exception if a device with that name does not exist</returns>
		const cudaDeviceProp& get_device_property(const std::string& device_name) const 
		{
			return  get_device_info(device_name).first;
		}

		/// <summary>
		/// Get device number
		/// </summary>
		/// <param name="device_name">the name of the device</param>
		/// <returns>device properties or throws an exception if a device with that name does not exist</returns>
		int get_device_number(const std::string& device_name) const
		{
			return  get_device_info(device_name).second;
		}

		/// <summary>
		/// Get the number of installed devices
		/// </summary>
		/// <returns>a number which represents the number of installed nvidia gpu</returns>
		size_t get_number_of_installed_devices() const
		{
			return installed_devices_.size();
		}
	};

	/// <summary>
	/// This method check the last error
	/// </summary>
	/// <param name="error">optional parameter, if not specified, the cudaGetLastErrorMethod will be called</param>
	static void check(const cudaError_t& error = cudaSuccess)
	{
		std::stringstream stream;

		//check the error
		if (error != cudaSuccess)
		{
			stream << "Error: " << cudaGetErrorString(error);
			throw std::exception(stream.str().c_str());
		}

		//if there is not error, check the last error
		cudaError_t last_error{};
		if ((last_error = cudaGetLastError()) == cudaSuccess)
		{
			return;
		}

		stream << "Error: " << cudaGetErrorString(error);
		throw std::exception(stream.str().c_str());
	}

	static device_interactions get_cuda_device_interactions()
	{
		//get the number of devices
		int device_no{};
		check(cudaGetDeviceCount(&device_no));

		//iterate through devices
		device_interactions device_interactions{};
		for (auto device = 0; device < device_no; ++device)
		{
			//get properties for a specific device
			cudaDeviceProp prop{};
			check(cudaGetDeviceProperties(&prop, device));

			//add device to device list
			device_interactions.add_device(prop.name, std::make_pair(prop, device));
		}

		//return device properties
		return device_interactions;
	}
};

