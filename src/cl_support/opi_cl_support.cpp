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
#include "opi_cl_support.h"

ClSupportImpl::ClSupportImpl()
{
	CUDAProperties = 0;
}

ClSupportImpl::~ClSupportImpl()
{
	delete[] CUDAProperties;
}

void ClSupportImpl::init()
{
	std::cout << "Calling CL init function" << std::endl;
	std::vector<cl::Platform> platformList;
	cl::Platform::get(&platformList);
	std::string vendor;
	std::string name;
	std::string ext;
	std::string dev;
	cl_int error;
	for (int i = 0; i < platformList.size(); i++) {
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VENDOR, &vendor);
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_NAME, &name);
		platformList[i].getInfo((cl_platform_info)CL_PLATFORM_VERSION, &ext);
		std::cout << "Platform " << i << ": " << vendor << " " << name << ": " << ext << std::endl;
		platformList[i].getDevices(CL_DEVICE_TYPE_ALL, &deviceList);
		for (int j = 0; j < deviceList.size(); j++) {
			deviceList[j].getInfo((cl_device_info)CL_DEVICE_NAME, &dev);
			std::cout << "Device " << j << ": " << dev << std::endl;
		}
	}

	//Just select the first platform and device for now
	cl_context_properties cprops[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platformList[0])(), 0 };
	context = cl::Context(CL_DEVICE_TYPE_ALL, cprops, NULL, NULL, &error);
	if (error != CL_SUCCESS) std::cerr << "Error: " << error << std::endl;

	currentDevice = 0;
	deviceList = context.getInfo<CL_CONTEXT_DEVICES>();
	defaultQueue = cl::CommandQueue(context, deviceList[currentDevice], 0, &error);

	/*
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
	*/
}

void ClSupportImpl::allocate(void** a, size_t size)
{
	*a = new cl::Buffer(context, CL_MEM_READ_WRITE, size);
}

void ClSupportImpl::free(void *mem)
{
	delete(mem);
}

void ClSupportImpl::copy(void *destination, void *source, size_t size, bool host_to_device)
{
	if (host_to_device) {
		defaultQueue.enqueueWriteBuffer(*static_cast<cl::Buffer*>(destination), CL_TRUE, 0, size, source);
	}
	else {
		defaultQueue.enqueueReadBuffer(*static_cast<cl::Buffer*>(source), CL_TRUE, 0, size, destination);
	}
}

void ClSupportImpl::shutdown()
{
	//cudaThreadExit();
}

void ClSupportImpl::selectDevice(int device)
{
	if (device < deviceList.size()) {
		currentDevice = device;
		defaultQueue = cl::CommandQueue(context, deviceList[currentDevice], 0);
	}
	else std::cout << "Invalid OpenCL device number - please select a number between 0 and "
		<< deviceList.size() - 1 << "." << std::endl;	
}

int ClSupportImpl::getCurrentDevice()
{
	return currentDevice;
}

int ClSupportImpl::getDeviceCount()
{
	return deviceList.size();
}

cudaDeviceProp* ClSupportImpl::getDeviceProperties(int device)
{
	/*
	if((device >= 0) && (device < getDeviceCount()))
		 return &CUDAProperties[device];
	*/
	return 0;
}

std::string ClSupportImpl::getCurrentDeviceName()
{
	std::string vendor, name;
	cl_device_type type;
	deviceList[currentDevice].getInfo(CL_DEVICE_VENDOR, &vendor);
	deviceList[currentDevice].getInfo(CL_DEVICE_NAME, &name);
	deviceList[currentDevice].getInfo(CL_DEVICE_TYPE, &type);
	std::stringstream result;
	result << vendor << " " << name << " ";
	if (type == CL_DEVICE_TYPE_GPU) { result << "(GPU)"; }
	else if (type == CL_DEVICE_TYPE_CPU) { result << "(CPU)"; }
	else result << "(other)";
	return result.str();
}

int ClSupportImpl::getCurrentDeviceCapability()
{
	//char* version;
	//deviceList[currentDevice].getInfo(CL_DRIVER_VERSION, &version);
	//std::cout << version << std::endl;
	return 1;
}

cl::Context ClSupportImpl::getOpenCLContext()
{
	return context;
}
cl::CommandQueue ClSupportImpl::getOpenCLQueue()
{
	return defaultQueue;
}

cl::Device ClSupportImpl::getOpenCLDevice()
{
	return deviceList[currentDevice];
}

std::vector<cl::Device> ClSupportImpl::getOpenCLDeviceList()
{
	return deviceList;
}


extern "C"
{
#if WIN32
__declspec(dllexport)
#endif
OPI::GpuSupport* createGpuSupport()
{
	return new ClSupportImpl();
}
}
