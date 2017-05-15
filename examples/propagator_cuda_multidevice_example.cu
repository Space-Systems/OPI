#include <cuda.h>
#include "OPI/opi_cpp.h"

// some plugin information
#define OPI_PLUGIN_NAME "CUDA Multi-Device Example Propagator"
#define OPI_PLUGIN_AUTHOR "ILR TU BS"
#define OPI_PLUGIN_DESC "A simple test"
// the plugin version
#define OPI_PLUGIN_VERSION_MAJOR 0
#define OPI_PLUGIN_VERSION_MINOR 1
#define OPI_PLUGIN_VERSION_PATCH 0

__global__ void device_cuda_init(OPI::Orbit* orbit, size_t size)
{
	for(int i = 0; i < size; ++i)	{
		orbit[i].inclination = i;
	}
}

__global__ void device_cuda_init2(OPI::Orbit* orbit, size_t size)
{
	for(int i = 0; i < size; ++i) {
		orbit[i].semi_major_axis = orbit[i].arg_of_perigee * 4;
	}
}

class CudaMultideviceExample:
		public OPI::Propagator
{
	public:
        CudaMultideviceExample(OPI::Host& host)
		{
		}

        virtual ~CudaMultideviceExample()
		{

		}

        virtual OPI::ErrorCode runPropagation(OPI::Population& data, double julian_day, double dt )
		{

			int deviceCount;
			cudaGetDeviceCount(&deviceCount);
			if(deviceCount == 1)
			{
				OPI::Orbit* orbit = data.getOrbit(OPI::DEVICE_CUDA + 0);
				device_cuda_init<<<dim3(1), dim3(1)>>>(orbit, data.getSize());
				device_cuda_init2<<<dim3(1), dim3(1)>>>(orbit, data.getSize());
				data.update(OPI::DATA_ORBIT, OPI::DEVICE_CUDA + 0);
			}
			else if(deviceCount > 1)
			{
				OPI::Orbit* orbit = data.getOrbit(OPI::DEVICE_CUDA + 0);
				cudaSetDevice(0);
				device_cuda_init<<<dim3(1), dim3(1)>>>(orbit, data.getSize());
				data.update(OPI::DATA_ORBIT, OPI::DEVICE_CUDA + 0);
				orbit = data.getOrbit(OPI::DEVICE_CUDA + 1);
				cudaSetDevice(1);
				device_cuda_init2<<<dim3(1), dim3(1)>>>(orbit, data.getSize());
				data.update(OPI::DATA_ORBIT, OPI::DEVICE_CUDA + 1);
			}
			else
				return OPI::CUDA_REQUIRED;
			return OPI::SUCCESS;
		}

		int requiresCUDA() {
			return 3;
		}

        int requiresOpenCL() {
            return 0;
        }

        int minimumOPIVersionRequired() {
            return 1;
        }

};

#define OPI_IMPLEMENT_CPP_PROPAGATOR CudaMultideviceExample

#include "OPI/opi_implement_plugin.h"

