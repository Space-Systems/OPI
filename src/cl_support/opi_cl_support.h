#include "../OPI/opi_gpusupport.h"

#include <CL/cl.h>

#include <iostream>
#include <sstream>
#include <stdlib.h>

using namespace std;

class ClSupportImpl :
	public OPI::GpuSupport
{
public:
	ClSupportImpl();
	~ClSupportImpl();

    virtual void init(int platformNumber = 0, int deviceNumber = 0);

    virtual void copy(void* a, void* b, size_t size, unsigned int num_objects, bool host_to_device);
	virtual void allocate(void** a, size_t size);
	virtual void free(void* mem);
	virtual void shutdown();
	virtual void selectDevice(int device);
	virtual int getCurrentDevice();
    virtual const char* getCurrentDeviceName();
	virtual int getCurrentDeviceCapability();
	virtual int getDeviceCount();

    virtual cudaDeviceProp* getDeviceProperties(int device) { return NULL; }

    virtual cl_context* getOpenCLContext();
    virtual cl_command_queue* getOpenCLQueue();
    virtual cl_device_id* getOpenCLDevice();
    virtual cl_device_id** getOpenCLDeviceList();

private:
    cl_context context;
	cl_command_queue defaultQueue;
	cl_device_id* devices;
	cl_uint nDevices;
	int currentDevice;
};
