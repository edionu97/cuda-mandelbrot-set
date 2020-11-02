#pragma once

#include "../config/config_model/config.h"
#include "../cuda_helpers/cuda_data_convertor.h"
#include "../image/components/rgb.h"

namespace helpers
{
	/// <summary>
	/// This structure contains all the data that will be sent to the compute_mandelbrot_set for the image generation
	/// </summary>
	struct image_generation_data
	{
		rgb* color_map;
		unsigned char* image;

		size_t image_width;
		size_t image_height;
		size_t max_iterations;
	};
	
	//declare aliases
	typedef cuda_data_convertor<rgb>::device_data_wrapper device_rgb;
	typedef cuda_data_convertor<unsigned char>::device_data_wrapper device_image;

	/// <summary>
	/// This method it is used for getting device data
	/// </summary>
	/// <param name="device_rgb">the device rgb</param>
	/// <param name="device_image">the device image</param>
	/// <param name="config">the json config</param>
	/// <returns>an wrapper over the allocated data on the gpu</returns>
	__host__
	inline image_generation_data get_device_data(const device_rgb& device_rgb, const device_image& device_image, const config& config)
	{
		image_generation_data image_generation_data{};
		
		//set the pointers to both the color map and the image
		image_generation_data.color_map = device_rgb.get_device_data();
		image_generation_data.image = device_image.get_device_data();
		
		//set the other information required
		image_generation_data.max_iterations = config.max_iterations;
		image_generation_data.image_height = config.image_height;
		image_generation_data.image_width = config.image_width;

		//return the image generation data
		return image_generation_data;
	}

	/// <summary>
	/// This method it is used for preparing the lunch configuration
	/// </summary>
	/// <param name="properties">gpu properties</param>
	/// <param name="width">the image width</param>
	/// <param name="height">image height</param>
	/// <returns>a pair of elements representing the number of blocks and the number of threads (as a dim3 structure)</returns>
	__host__
	inline std::pair<dim3, dim3> prepare_launch_configuration(const cudaDeviceProp& properties, const size_t width, const size_t height)
	{
		//threads per block
		//get the threads per block size
		const auto max_block_size = static_cast<int>(sqrt(properties.maxThreadsPerBlock));
		const auto threads_per_block = dim3(max_block_size, max_block_size);

		//create the blocks 
		const auto blocks_rows = static_cast<int>(ceil((.0 + height) / max_block_size));
		const auto blocks_columns = static_cast<int>(ceil((.0 + width) / max_block_size));
		const auto blocks_size = dim3(blocks_rows, blocks_columns);

		//create the block size
		return { blocks_size, threads_per_block };
	}
};

