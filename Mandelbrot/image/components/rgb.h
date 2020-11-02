#pragma once
#include <algorithm>

struct rgb
{
	unsigned char red, green, blue;

	/// <summary>
	/// Create the rgb component
	/// </summary>
	/// <param name="red_value">the red color value</param>
	/// <param name="green_value">the green color value</param>
	/// <param name="blue_value">the blue color</param>
	rgb(const double red_value, const double green_value, const double blue_value)
	{
		red = convert_color_value_to_byte(red_value);
		green = convert_color_value_to_byte(green_value);
		blue = convert_color_value_to_byte(blue_value);
	}

private:

	/// <summary>
	/// This method converts the value to color byte ensuring that the value is between 0 and 255  
	/// </summary>
	/// <param name="value">the color value</param>
	/// <returns></returns>
	static unsigned char convert_color_value_to_byte(const double value)
	{
		//multiply the value with 255
		auto converted_value = value * 255;

		//ensure that the value is between 0 and 255 (byte value)
		converted_value = std::max<double>(0, converted_value);
		converted_value = std::min<double>(255, converted_value);

		//convert the value to byte
		return static_cast<unsigned char>(converted_value);
	}
};