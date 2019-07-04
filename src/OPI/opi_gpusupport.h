#ifndef OPI_GPU_SUPPORT_H
#define OPI_GPU_SUPPORT_H
#include <iostream>

#ifndef OPI_DISABLE_OPENCL
#include <CL/cl.h>
#endif

struct cudaDeviceProp;

namespace OPI
{
	class GpuSupport;

	typedef GpuSupport* (*procCreateGpuSupport)();
	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */

	/**
	 * @ingroup CPP_API_GROUP
	 * @brief CUDA/OpenCL Support Interface
	 */
	class GpuSupport
	{
		public:
			virtual ~GpuSupport() {}
            virtual void init(int platformNumber = 0, int deviceNumber = 0) = 0;

			virtual void allocate(void** a, size_t size) = 0;
			virtual void free(void* mem) = 0;
            virtual void copy(void* dest, void* source, size_t size, unsigned int num_objects, bool host_to_device) = 0;

			virtual void shutdown() = 0;

			virtual void selectDevice(int device) = 0;
			virtual int getCurrentDevice() = 0;
			virtual int getCurrentDeviceCapability() = 0;
            virtual const char* getCurrentDeviceName() = 0;

			virtual int getDeviceCount() = 0;

            virtual cudaDeviceProp* getDeviceProperties(int device) = 0;

#ifndef OPI_DISABLE_OPENCL
            virtual cl_context* getOpenCLContext() = 0;
            virtual cl_command_queue* getOpenCLQueue() = 0;
            virtual cl_device_id* getOpenCLDevice() = 0;
            virtual cl_device_id** getOpenCLDeviceList() = 0;
#endif

		protected:

	};
	/**
	 * \endcond INTERNAL_DOCUMENTATION
	 */
}


#endif
