#pragma once

#include "cuda_runtime.h"

#include "device_launch_parameters.h"

#include "../utils/helpers.h"

#define MIN(a,b) (a) < (b) ? (a) : (b); 

/// <summary>
/// This method it is used for getting the iteration number for a given set of data
/// </summary>
/// <param name="width">the image width</param>
/// <param name="height">the image height</param>
/// <param name="max_iterations">the number of max iterations</param>
/// <param name="row">the row</param>
/// <param name="col">the column</param>
/// <returns>a value that represents the max iteration number</returns>
__device__
inline size_t get_iterations_number(const double width,
									const double height,
									const size_t max_iterations, const int row, const int col)
{
	//create the real and imaginary part
	const auto c_real_part = (col - width / 2.0) * 4.0 / width;
	const auto c_imaginary_part = (row - height / 2.0) * 4.0 / width;

	//iterate fc function
	size_t iteration{};
	for (double x = 0, y = 0; x * x + y * y <= 4 && iteration < max_iterations; ++iteration)
	{
		const auto x_new = x * x - y * y + c_real_part;
		y = 2 * x * y + c_imaginary_part;
		x = x_new;
	}

	//get the min value between iteration and max iteration
	return MIN(iteration, max_iterations);
}

/// <summary>
/// This function is used for setting a pixel for a specific row and column
/// </summary>
/// <param name="image">the image into that we want to set the pixel</param>
/// <param name="width">the image width</param>
/// <param name="x">column</param>
/// <param name="y">row</param>
/// <param name="pixel_value">the value that will be set to pixel</param>
__device__
inline void set_image_pixel(unsigned char* image, 
						    const unsigned int width,
							const unsigned int x,
							const unsigned int y,
							const rgb& pixel_value)
{
	image[4 * width * y + x * 4 + 0] = pixel_value.red;
	image[4 * width * y + x * 4 + 1] = pixel_value.green;
	image[4 * width * y + x * 4 + 2] = pixel_value.blue;
	image[4 * width * y + x * 4 + 3] = 255;
}

/// <summary>
/// This method represents the kernel of the program and it is used for computing the mandelbrot set
/// </summary>
/// <param name="device_data">
///		All the data used for required computations, since cuda 6.x we can structure by value into the kernel (because of the unified memory introduction)
///		See: https://developer.nvidia.com/blog/unified-memory-in-cuda-6/
/// </param>
__global__
inline void compute_mandelbrot_set(const helpers::image_generation_data device_data)
{
	//get the row and the column
	const auto row = threadIdx.x + blockIdx.x * blockDim.x;
	const auto col = threadIdx.y + blockIdx.y * blockDim.y;

	//check that we are still into the bounds
	if(!(row < device_data.image_height && col < device_data.image_width))
	{
		return;
	}

	//get iterations number
	const auto iterations_number = get_iterations_number(
		static_cast<double>(device_data.image_width),
		static_cast<double>(device_data.image_height),
		device_data.max_iterations,
		row, 
		col
	);

	//set the pixel, with the corresponding values
	set_image_pixel(
		device_data.image, 
		static_cast<unsigned int>(device_data.image_width), 
		col,
		row, 
		device_data.color_map[iterations_number]);
}