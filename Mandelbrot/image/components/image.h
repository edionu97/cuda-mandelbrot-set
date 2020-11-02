#pragma once

#include <vector>

class image
{
	size_t width_;
	size_t height_;
	const std::vector<unsigned char>& image_bytes_;

public:

	explicit image(
		const size_t width,
		const size_t height,
		const std::vector<unsigned char>& bytes) : width_{ width },
											 height_{ height },
											 image_bytes_{ bytes }
	{
	}

	image(image&& other) noexcept
		: width_(other.width_),
		height_(other.height_),
		image_bytes_(other.image_bytes_)
	{
	}


	image(const image& other) = delete;
	image& operator=(const image& other) = delete;
	image& operator=(image&& other) noexcept = delete;

	size_t get_width() const
	{
		return  width_;
	}

	size_t get_height() const
	{
		return  height_;
	}

	const std::vector<unsigned char>& get_image_bytes() const
	{
		return  image_bytes_;
	}

	~image() = default;
};
