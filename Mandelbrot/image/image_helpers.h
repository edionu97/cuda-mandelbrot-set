#pragma once
#include <lodepng.h>
#include <fstream>

#include "components/rgb.h"
#include "../config/config_model/config.h"
#include "components/image.h"


class image_helpers
{
	/// <summary>
	/// convert hsv to rgb
	/// </summary>
	/// <param name="hue">the hue</param>
	/// <param name="saturation">saturation</param>
	/// <param name="value">the value</param>
	/// <returns>a new instance of rgb</returns>
	static rgb convert_hsv_to_rgb(double hue, const double saturation, const double value)
	{
		hue /= 60;

		const auto i = static_cast<int>(floor(hue));
		const auto f = hue - i;

		const auto p = value * (1 - saturation);
		const auto q = value * (1 - saturation * f);
		const auto t = value * (1 - saturation * (1 - f));

		//get the proper values for red,green and blue component of the image
		double r{}, g{}, b{};
		switch (i)
		{
			case 0:
			{
				r = value;
				g = t;
				b = p;
				break;
			}
			case 1:
			{
				r = q;
				g = value;
				b = p;
				break;
			}
			case 2:
			{

				r = p;
				g = value;
				b = t;
				break;
			}
			case 3:
			{
				r = p;
				g = q;
				b = value;
				break;
			}
			case 4:
			{
				r = t;
				g = p;
				b = value;
				break;
			}
			default:
			{
				r = value;
				g = p;
				b = q;
				break;
			}
		}

		//create a rgb instance
		return { r,g,b };
	}

	static void check(const unsigned png_error)
	{
		if (png_error == 0)
		{
			return;
		}

		throw std::exception(lodepng_error_text(png_error));
	}

public:

	/// <summary>
	/// This method is used for creating a color map with a specific size
	/// That map contains information about pixels
	/// Example for pixel i the rgb values will be located at 3 * i, 3 * i + 1,3 * i + 2
	/// </summary>
	/// <param name="map_size"></param>
	/// <returns></returns>
	__host__
	static std::vector<rgb> create_color_map(const unsigned int map_size)
	{
		//declare the array
		std::vector<rgb> color_map;

		//iterate through items
		for (unsigned int index = 0; index < map_size; ++index)
		{
			//create hsv
			const auto hue = (index + .0) / 4.0;
			const auto saturation = 1.0;
			const auto value = (index + .0) / (index + 8.0);

			//convert hsv to rgb
			const auto rgb = convert_hsv_to_rgb(hue, saturation, value);

			//push the elements into 
			color_map.push_back(rgb);
		}

		//put the last values
		color_map.emplace_back(0, 0, 0);

		//return the color map
		return color_map;
	}

	/// <summary>
	/// Save the image into file
	/// </summary>
	/// <param name="image">the image</param>
	/// <param name="configuration">the configuration</param>
	__host__
	static void save_image_to_file(const image& image, const config& configuration)
	{
		//intize the state
		LodePNGState state;
		lodepng_state_init(&state);

		unsigned char* png_image = nullptr;
		size_t png_image_size{};

		try
		{
			//encode the image
			check(lodepng_encode(
				&png_image,
				&png_image_size,
				&image.get_image_bytes()[0],
				static_cast<unsigned int>(image.get_width()),
				static_cast<unsigned int>(image.get_height()), &state));

			//create the file
			std::ofstream out(configuration.location.c_str());

			//save the image
			check(lodepng_save_file(png_image, png_image_size, configuration.location.c_str()));

			//execute the cleanup
			lodepng_state_cleanup(&state);
			if (png_image != nullptr)
			{
				free(png_image);
			}
		}
		catch (std::exception& e)
		{
			//execute the cleanup
			lodepng_state_cleanup(&state);

			if (png_image != nullptr)
			{
				free(png_image);
			}

			throw e;
		}
	}
};

