#include "dynlib.h"
#include <iostream>
namespace OPI
{

	DynLib::DynLib(const std::string& name, bool silentfail )
	{
		handle = 0;
#ifdef _WIN32
		std::string buffer = name;
		handle = LoadLibrary(buffer.c_str());
		if((!handle) && (!silentfail))
		{
			std::cout << "Error loading lib" << std::endl;
		}
#else
		std::string buffer = name;
		handle = dlopen(buffer.c_str(), RTLD_LAZY);
		if((!handle) && (!silentfail))
		{
			std::cout << dlerror() << std::endl;
		}
#endif
	}

	void* DynLib::loadFunction(const char *name, bool silentfail)
	{
		if(handle == 0)
			return 0;
#ifdef _WIN32
		void* result = 0;
		result = (void*)GetProcAddress(handle, name);
		if(!result)
		{
			if(!silentfail)
				std::cout << "Error loading Function " << name << std::endl;
		}
#else
		void* result = 0;
		char* error;
		result = dlsym(handle, name);
		if((error = dlerror()) != NULL)
		{
			if(!silentfail)
				std::cout << "Error loading Function " << name << ": " << error << std::endl;
		}
#endif
		return result;
	}

	bool DynLib::isValid() const
	{
		return (handle != 0);
	}

	std::string DynLib::getSuffix()
	{
#ifdef _WIN32
		return ".dll";
#else
		return ".so";
#endif
	}

	DynLib::~DynLib()
	{
		if(handle != 0)
		{
#ifdef _WIN32
			FreeLibrary(handle);
#else
			dlclose(handle);
#endif
		}
	}
}
