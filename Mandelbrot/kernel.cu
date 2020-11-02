#include <iostream>

#include "cuda_helpers/cuda_data_convertor.h"
#include "cuda_helpers/cuda_device_helpers.h"
#include "image/image_helpers.h"
#include "resources/resources_manager.h"
#include "utils/helpers.h"
#include "cuda_helpers/cuda_utils.h"

#include "gpu/mandelbrot.cuh"

using namespace cuda_print_utils;

/// <summary>
/// This method it is used for generating the mandelbrot set on the gpu
/// </summary>
/// <param name="device_properties">the gpu information</param>
/// <param name="json_config">the json config</param>
/// <param name="time_computer">used for computing elapsed time</param>
/// <returns>image bytes</returns>
std::vector<unsigned char> generate_mandelbrot_image(const cudaDeviceProp& device_properties, 
													 const config& json_config, 
													 statistics::elapsed_time_computer& time_computer)
{
	//set the initial time period
	time_computer.set_time_period();
	
	//send data to device
	const auto color_map = image_helpers::create_color_map(json_config.max_iterations);
	const auto device_color_map = cuda_data_convertor<rgb>::convert_stl_vector_to_device_array(color_map);

	const std::vector<unsigned char> image_vector(json_config.image_height * json_config.image_width * 4, 0);
	const auto device_image = cuda_data_convertor<unsigned char>::convert_stl_vector_to_device_array(image_vector);

	const auto device_data = helpers::get_device_data(device_color_map, device_image, json_config);

	//prepare launch config
	auto lunch_configuration = helpers::prepare_launch_configuration(device_properties, json_config.image_width, json_config.image_height);

	time_computer.set_time_period("Copy data CPU to GPU took: ");
	
	compute_mandelbrot_set <<< lunch_configuration.first, lunch_configuration.second >>> (device_data);

	time_computer.set_time_period("GPU mandelbrot set computation took: ");

	auto image_bytes = cuda_data_convertor<unsigned char>::convert_device_data_to_stl_vector(device_image, image_vector.size());

	time_computer.set_time_period("Copy data from GPU to CPU took: ", true);
	
	return image_bytes;
}

int main()    // NOLINT(bugprone-exception-escape)
{
	try
	{
		//read the configuration
		const auto config = resources_manager::get_config();

		//get the cuda device interaction
		const auto cuda_device_interaction = cuda_device_helpers::get_cuda_device_interactions();

		//check if we have installed some nvidia gpu
		if (cuda_device_interaction.get_number_of_installed_devices() == 0)
		{
			throw std::exception("No Nvidia Gpu installed");
		}

		//get device properties
		const auto& device_properties = cuda_device_interaction.get_device_property(config.device_name);

		//print device properties
		std::cout << device_properties << '\n';

		statistics::elapsed_time_computer time_computer;
		
		//generate the mandelbrot image
		const auto image_bytes = generate_mandelbrot_image(device_properties, config, time_computer);

		//print time periods
		time_computer.print_time_periods(std::cout);

		//check for any error
		cuda_device_helpers::check();

		//create the image from the image bytes
		const image img{ config.image_width, config.image_height, image_bytes };

		//save the image
		image_helpers::save_image_to_file(img, config);
	}
	catch (std::exception& e)
	{
		std::cout << e.what() << '\n';
	}

	return 0;
}

