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
#include "../OPI/opi_gpusupport.h"

#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdlib.h>

using namespace std;

class CudaSupportImpl:
		public OPI::GpuSupport
{
	public:
		CudaSupportImpl();
		~CudaSupportImpl();

		virtual void init();

        virtual void copy(void* a, void* b, size_t size, unsigned int num_objects, bool host_to_device);
		virtual void allocate(void** a, size_t size);
		virtual void free(void* mem);
		virtual void shutdown();
		virtual void selectDevice(int device);
		virtual int getCurrentDevice();
		virtual std::string getCurrentDeviceName();
		virtual int getCurrentDeviceCapability();
		virtual int getDeviceCount();
        virtual cudaDeviceProp* getDeviceProperties(int device);
#ifndef OPI_DISABLE_OPENCL
        virtual cl_context* getOpenCLContext() { return NULL; }
        virtual cl_command_queue* getOpenCLQueue() { return NULL; }
        virtual cl_device_id* getOpenCLDevice() { return NULL; }
        virtual cl_device_id** getOpenCLDeviceList() { return NULL; }
#endif
	private:
		cudaDeviceProp* CUDAProperties;
};

CudaSupportImpl::CudaSupportImpl()
{
	CUDAProperties = 0;
}

CudaSupportImpl::~CudaSupportImpl()
{
	delete[] CUDAProperties;
}

void CudaSupportImpl::init()
{
	int deviceCount = 0;
	int deviceNumber = 0;

	// search for devices and print some information
	// currently, only the first device is used
	cudaGetDeviceCount(&deviceCount);
	if (deviceCount == 0) {
		cout << "  No CUDA-capable devices found." << endl;
	}
	else {
		CUDAProperties = new cudaDeviceProp[deviceCount];
		cout << "  Found " << deviceCount << " CUDA capable device(s): " << endl << endl;
		for (int i=0; i<deviceCount; i++) {
			cudaGetDeviceProperties(&(CUDAProperties[i]), i);
			cudaDeviceProp& deviceProp = CUDAProperties[i];
			int tpb = deviceProp.maxThreadsPerBlock;
			int bs[3];
			int gs[3];
			for (int j=0; j<3; j++) {
				bs[j] = deviceProp.maxThreadsDim[j];
				gs[j] = deviceProp.maxGridSize[j];
			}
			cout << "  Device Number:      " << i << endl;
			cout << "  Name:               " << deviceProp.name << endl;
			cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << endl;
			cout << "  Total Memory:       " << (deviceProp.totalGlobalMem/(1024*1024)) << "MB" << endl;
			cout << "  Clock Speed:        " << (deviceProp.clockRate/1000) << "MHz" << endl;
			cout << "  Threads per Block:  " << tpb << endl;
			cout << "  Block Dimensions:   " << bs[0] << "/" << bs[1] << "/" << bs[2] << endl;
			cout << "  Grid Dimensions:    " << gs[0] << "/" << gs[1] << "/" << gs[2] << endl;
			cout << "  Warp Size:          " << deviceProp.warpSize << endl;
			cout << "  MP Count:           " << deviceProp.multiProcessorCount << endl;
			cout << endl;
		}

		deviceNumber = 0;

		cudaSetDevice(deviceNumber);
	}
}

void CudaSupportImpl::allocate(void** a, size_t size)
{
	cudaMalloc(a, size);
}

void CudaSupportImpl::free(void *mem)
{
	cudaFree(mem);
}

void CudaSupportImpl::copy(void *destination, void *source, size_t size, unsigned int num_objects, bool host_to_device)
{
    cudaMemcpy(destination, source, size*num_objects, host_to_device ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost);
}

void CudaSupportImpl::shutdown()
{
	cudaThreadExit();
}

void CudaSupportImpl::selectDevice(int device)
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);
	if( device < deviceCount) {
		cudaSetDevice(device);
	}
}

int CudaSupportImpl::getCurrentDevice()
{
	int device = -1;
	cudaGetDevice(&device);
	return device;
}

int CudaSupportImpl::getDeviceCount()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	return deviceCount;
}

cudaDeviceProp* CudaSupportImpl::getDeviceProperties(int device)
{
	if((device >= 0) && (device < getDeviceCount()))
		 return &CUDAProperties[device];
	return 0;
}

std::string CudaSupportImpl::getCurrentDeviceName()
{
	int device = getCurrentDevice();
	if((device >= 0) && (device < getDeviceCount())) {
		// Build the return string. Currently, all CUDA devices
		// are made by Nvidia, there seems to be no data field
		// for a manufacturer's name in the properties struct.
		std::stringstream result;
		result << "NVIDIA ";
		result << (const char*)&CUDAProperties[device].name;
		result << " @ " << (CUDAProperties[device].clockRate)/1000 << "MHz";
		if (&CUDAProperties[device].ECCEnabled)
			result << " (ECC enabled)";
		return result.str();
	}
	else {
		return std::string("No CUDA Device selected.");
	}
}

int CudaSupportImpl::getCurrentDeviceCapability()
{
	int device = getCurrentDevice();
	if((device >= 0) && (device < getDeviceCount())) {
		int major = CUDAProperties[device].major;
		return major;
	}
	else {
		// No device selected.
		return -1;
	}
}

extern "C"
{
#if WIN32
__declspec(dllexport)
#endif
OPI::GpuSupport* createGpuSupport()
{
	return new CudaSupportImpl();
}
}
