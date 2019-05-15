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
	
}

ClSupportImpl::~ClSupportImpl()
{
	
}

void ClSupportImpl::init(int platformNumber, int deviceNumber)
{
	std::cout << "Calling CL init function" << std::endl;
	cl_platform_id platforms[8];
	cl_uint nPlatforms;
	cl_int error;
	error = clGetPlatformIDs(8, platforms, &nPlatforms);
    if (error == CL_SUCCESS)
    {
        for (unsigned int i = 0; i < nPlatforms; i++) {
            size_t actualLength;
            char vendor[32], name[64];
            clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 32, &vendor, &actualLength);
            clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,   64, &name,   &actualLength);
            std::cout << "Platform " << i << ": " << string(vendor) << " " << string(name) << std::endl;

            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &nDevices);
            devices = new cl_device_id[nDevices];
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, nDevices, devices, &nDevices);
            for (unsigned int j = 0; j < nDevices; j++) {
                char devName[64];
                clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 64, devName, &actualLength);
                std::cout << "Device " << j << ": " << string(devName) << std::endl;
            }
        }

        //FIXME Check availability
        int currentPlatform = platformNumber;
        currentDevice = deviceNumber;

        clGetDeviceIDs(platforms[currentPlatform], CL_DEVICE_TYPE_ALL, nDevices, devices, &nDevices);
        cl_context_properties props[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[currentPlatform]), 0 };
        context = clCreateContext(props, 1, devices, NULL, NULL, &error);
        if (error != CL_SUCCESS) std::cerr << "Error creating context: " << error << std::endl;
        defaultQueue = clCreateCommandQueue(context, devices[currentDevice], 0, &error);
        if (error != CL_SUCCESS) std::cerr << "Error creating queue: " << error << std::endl;
    }
    else {
        std::cerr << "Unable to get OpenCL platform IDs: " << error << std::endl;
    }
}

void ClSupportImpl::allocate(void** a, size_t size)
{
	cl_int error;
	*a = clCreateBuffer(context, CL_MEM_READ_WRITE, size, NULL, &error);
	if (error != CL_SUCCESS) std::cout << "Error allocating OpenCL buffer memory: " << error << std::endl;
    //else cout << "Allocated " << size << " bytes at " << *a << endl;
}

void ClSupportImpl::free(void *mem)
{
    cl_int error = clReleaseMemObject(static_cast<cl_mem>(mem));
    if (error != CL_SUCCESS) std::cout << "Error freeing OpenCL buffer memory: " << error << std::endl;
    //else cout << "Freed memory object at " << static_cast<cl_mem>(mem) << endl;
}

void ClSupportImpl::copy(void *destination, void *source, size_t size, unsigned int num_objects, bool host_to_device)
{
	cl_int error = CL_SUCCESS;
	if (host_to_device) {
		cl_mem destinationBuffer = static_cast<cl_mem>(destination);
        error = clEnqueueWriteBuffer(defaultQueue, destinationBuffer, CL_TRUE, 0, size*num_objects, source, 0, NULL, NULL);
		if (error != CL_SUCCESS) std::cout << "Error copying Population data to OpenCL device: " << error << std::endl;
        //else cout << "Copied " << size << " bytes to device at " << destinationBuffer << endl;
	}
	else {
		cl_mem sourceBuffer = static_cast<cl_mem>(source);
        error = clEnqueueReadBuffer(defaultQueue, sourceBuffer, CL_TRUE, 0, size*num_objects, destination, 0, NULL, NULL);
		if (error != CL_SUCCESS) std::cout << "Error downloading Population data from OpenCL device: " << error << std::endl;
        //else cout << "Downloaded " << size << " bytes from device (" << sourceBuffer << " to " << destination << ")" << endl;
    }
}

void ClSupportImpl::shutdown()
{
	clReleaseCommandQueue(defaultQueue);
	clReleaseContext(context);
}

void ClSupportImpl::selectDevice(int device)
{
    if (device != currentDevice)
    {
        if (device < nDevices) {
            //FIXME: Device switching not yet supported
        }
        else std::cout << "Invalid OpenCL device number - please select a number between 0 and "
            << nDevices - 1 << "." << std::endl;
    }
}

int ClSupportImpl::getCurrentDevice()
{
	return currentDevice;
}

int ClSupportImpl::getDeviceCount()
{
	return nDevices;
}

const char* ClSupportImpl::getCurrentDeviceName()
{
	/*
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
	return result.str();*/
	return "NOT YET IMPLEMENTED";
}

int ClSupportImpl::getCurrentDeviceCapability()
{
	//char* version;
	//deviceList[currentDevice].getInfo(CL_DRIVER_VERSION, &version);
	//std::cout << version << std::endl;
	return 1;
}

cl_context* ClSupportImpl::getOpenCLContext()
{
    return &context;
}

cl_command_queue* ClSupportImpl::getOpenCLQueue()
{
    return &defaultQueue;
}

cl_device_id* ClSupportImpl::getOpenCLDevice()
{
    return &devices[currentDevice];
}

cl_device_id** ClSupportImpl::getOpenCLDeviceList()
{
    return &devices;
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
