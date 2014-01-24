#include <iostream>
#include <iomanip>
#include <stdint.h>
#include <time.h>
#include <sys/timeb.h>
#include <cstdlib>
#include "OPI/opi_cpp.h"
#include <fstream>

// needs -lrt (real-time lib)
// 1970-01-01 epoch UTC time, 1 mcs resolution (divide by 1M to get time_t)
uint64_t getTime()
{
		timespec ts;
		clock_gettime(CLOCK_REALTIME, &ts);
		return (uint64_t)ts.tv_sec * 1000000LL + (uint64_t)ts.tv_nsec / 1000LL;
}
int main(int argc, char* argv[])
{
	if(argc != 9) {
		std::cout << "Usage:" << std::endl;
		std::cout << "  benchmark <plugin_dir> <data_file> <propagator> <query_plugin> <iterations> <start_time> <dt> <result>" << std::endl;
	}
	else {
		OPI::Host host;
		host.loadPlugins(argv[1]);

		OPI::ObjectData data(host);

		data.read(argv[2]);

		OPI::Propagator* prop = host.getPropagator(argv[3]);
		OPI::DistanceQuery* query = host.getDistanceQuery(argv[4]);
		int iterations = atoi(argv[5]);
		int start_time = atoi(argv[6]);
		int dt = atoi(argv[7]);
		uint64_t start;
		uint64_t end;

		std::ofstream file(argv[8]);

		uint64_t average_propagation = 0;
		uint64_t average_rebuild = 0;
		uint64_t average_query = 0;

		if(prop && query)
		{
			file << "# OPI Benchmark results " << std::endl;
			file << "# Plugin dir: " << argv[1] << std::endl;
			file << "# Input data: " << argv[2] << std::endl;
			file << "# Propagator: " << argv[3] << std::endl;
			file << "# Query: " << argv[4] << std::endl;
			file << "# Iterations: " << iterations << std::endl;
			file << "# Starting time: " << start_time << std::endl;
			file << "# dt: " << dt << std::endl;
			file << "# Population size: " << data.getSize() << std::endl;

			start = getTime();
			OPI::Orbit* orbit = data.getOrbit(OPI::DEVICE_CUDA);
			OPI::ObjectProperties* props = data.getObjectProperties(OPI::DEVICE_CUDA);
			end = getTime();
			file << "# Initial cuda synchronisation time: " << (end - start) << std::endl;
			file << "# time_propagation time_rebuild time_query collision_count" << std::endl;
			for(int i = 0; i < iterations; ++i)
			{
				std::cout << "Running iteration " << i << " of " << iterations <<
										 " using " << argv[3] << " and " << argv[4] << " on " << argv[2] << std::endl;
				start = getTime();
				prop->propagate(data, 0., start_time + i * dt, dt);
				end = getTime();
				average_propagation += end - start;
				file << std::setw(8) << (end - start) << " ";

				start = getTime();
				query->rebuild(data);
				end = getTime();
				average_rebuild += end - start;
				file << std::setw(8) << (end - start) << " ";

				OPI::IndexPairList pairs(host);

				pairs.reserve(4000000);

				start = getTime();
				query->queryCubicPairs(data, pairs, 10);
				end = getTime();
				average_query += end - start;
				file << std::setw(8) << (end - start) << " ";

				file << pairs.getPairsUsed() << std::endl;
			}

			file << "# avg_prop/avg_rebuild/avg_query [ms]" << std::endl;
			file << "# " << (average_propagation / (float)(iterations * 1000))
					 << "/" << (average_rebuild / (float)(iterations * 1000))
					 << "/" << (average_query / (float)(iterations * 1000)) << std::endl;
		}
		else
		{
			std::cout << "Propator and/or Query Plugin not found!" << std::endl;
		}
	}
	return EXIT_SUCCESS;
}
