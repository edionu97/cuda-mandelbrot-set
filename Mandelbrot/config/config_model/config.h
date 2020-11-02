#pragma once
#include <string>
#include <nlohmann/json.hpp>

/// <summary>
/// Represents the json configuration mapped to an object
/// </summary>
struct config
{
	unsigned int max_iterations;
	
	size_t image_width;
	size_t image_height;
	
	std::string location;
	std::string device_name;

	/// <summary>
	/// Create the configuration from json aka parse the object
	/// </summary>
	/// <param name="json">the json object that contains data</param>
	explicit config(const nlohmann::json& json)
	{
		//parse the max_iterations
		max_iterations = json[max_iteration_mapping_tag_].get<unsigned int>();

		//parse the generated image part
		const auto& generate_image_json = json[generated_image_tag_];
		image_width = generate_image_json[image_width_tag_].get<size_t>();
		image_height = generate_image_json[image_height_tag_].get<size_t>();
		location = generate_image_json[location_tag_].get<std::string>();

		//parse cuda part
		const auto& cuda_json = json[cuda_tag_];
		device_name = cuda_json[device_name_tag_].get<std::string>();
	}
	
private:
	std::string max_iteration_mapping_tag_{ "maxIterations" };
	std::string generated_image_tag_{ "generatedImage" };
	std::string image_width_tag_{ "imageWidth" };
	std::string image_height_tag_{ "imageHeight" };
	std::string location_tag_{ "location" };
	std::string cuda_tag_{ "cuda" };
	std::string device_name_tag_{ "deviceName" };
};


