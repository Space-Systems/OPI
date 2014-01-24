#ifndef OPI_CUDA_SUPPORT_H
#define OPI_CUDA_SUPPORT_H
#include <iostream>
struct cudaDeviceProp;
namespace OPI
{
	class CudaSupport;

	typedef CudaSupport* (*procCreateCudaSupport)();
	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */

	/**
	 * @ingroup CPP_API_GROUP
	 * @brief CUDA Support Interface
	 */
	class CudaSupport
	{
		public:
			virtual ~CudaSupport() {}
			virtual void init() = 0;

			virtual void allocate(void** a, size_t size) = 0;
			virtual void free(void* mem) = 0;
			virtual void copy(void* dest, void* source, size_t size, bool host_to_device) = 0;

			virtual void shutdown() = 0;

			virtual void selectDevice(int device) = 0;
			virtual cudaDeviceProp* getDeviceProperties(int device) = 0;
			virtual int getCurrentDevice() = 0;

			virtual int getDeviceCount() = 0;
		protected:

	};
	/**
	 * \endcond INTERNAL_DOCUMENTATION
	 */
}


#endif
