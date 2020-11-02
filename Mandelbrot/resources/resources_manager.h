#pragma once

#include <nlohmann/json.hpp>
#include <fstream>

#include "../config/config_model/config.h"

class resources_manager
{
public:

	/// <summary>
	/// Read the configuration
	/// </summary>
	/// <returns>a new instance of config</returns>
	static config get_config()
	{
		//read the json from file
		nlohmann::json json;
		std::ifstream{ "config/config.json" } >> json;

		//return the config
		return config(json);
	}
};

