/* OPI: Orbital Propagation Interface
 * Copyright (C) 2014 Institute of Aerospace Systems, TU Braunschweig, All rights reserved.
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 3.0 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library.
 */
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
			virtual std::string getCurrentDeviceName() = 0;

			virtual int getDeviceCount() = 0;
		protected:

	};
	/**
	 * \endcond INTERNAL_DOCUMENTATION
	 */
}


#endif
