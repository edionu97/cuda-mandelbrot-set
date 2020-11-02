#pragma once
#include <driver_types.h>
#include <ostream>

namespace cuda_print_utils
{
	/// <summary>
	/// This method it is used for printing at the console the device properties
	/// </summary>
	/// <param name="out">the output stream</param>
	/// <param name="props">the device properties</param>
	/// <returns>the same stream, for operator chaining</returns>
	inline auto operator <<(std::ostream& out, const cudaDeviceProp& props) -> std::ostream&
	{
		//device generic information
		out << "Device name: " << props.name << '\n';
		out << "Dedicated GPU: " << std::boolalpha << !props.integrated << '\n';
		out << "Compute capability: " << props.major << '.' << props.minor << '\n';
		out << "Total memory: " << props.totalGlobalMem / 1020 / 1024 << '\n';

		//threading information
		out << "Max threads dim: " << props.maxThreadsDim[0] << " x ";
		out << props.maxThreadsDim[1] << " x ";
		out << props.maxThreadsDim[2] << '\n';

		out << "Max threads per block: " << props.maxThreadsPerBlock << '\n';
		out << "Warp size:" << props.warpSize << '\n';

		//return instance of the stream
		return  out;
	}
	
}

namespace statistics
{
	class elapsed_time_computer
	{
		std::vector<std::pair<cudaEvent_t, std::string>> cuda_events_;
		
	public:

		/// <summary>
		/// This method will start the event and will start recording
		/// </summary>
		/// <param name="event_label">the label of the event, by default is null</param>
		/// <param name="is_last_event"></param>
		void set_time_period(const std::string& event_label = "", const bool is_last_event = false)
		{
			//create the event
			cudaEvent_t event;
			cuda_device_helpers::check(cudaEventCreate(&event));

			//start event recording
			cuda_device_helpers::check(cudaEventRecord(event));

			if (is_last_event)
			{
				cudaEventSynchronize(event);
			}

			//push the events
			cuda_events_.emplace_back(event, event_label);
		}

		/// <summary>
		/// This method will get two consecutive events and will compute the elapsed time between them
		/// </summary>
		/// <param name="out">the stream into which we are writing</param>
		void print_time_periods(std::ostream& out = std::cout) const
		{
			for(size_t idx = 1; idx < cuda_events_.size(); ++idx)
			{
				//get the events
				const auto& start = cuda_events_[idx - 1];
				const auto& end = cuda_events_[idx];

				//compute the time
				auto elapsed_time = .0f;
				cudaEventElapsedTime(&elapsed_time, start.first, end.first);

				//print the duration
				out << end.second << elapsed_time / 1000 << "sec\n";
			}
		}

		~elapsed_time_computer() noexcept
		{
			for (const auto & cuda_event : cuda_events_)
			{
				cudaEventDestroy(cuda_event.first);
			}
		}
	};
}
