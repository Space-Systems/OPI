#pragma once
#ifndef OPI_DYNAMIC_LIBRARIES_H
#define OPI_DYNAMIC_LIBRARIES_H
#ifdef WIN32
#include <windows.h>
typedef HINSTANCE DynLibHandle;
#else
#include <dlfcn.h>
typedef void* DynLibHandle;
#endif
#include <string>
namespace OPI
{
	/**
	 * @brief A platform-independent wrapper for shared object loading
	 * @ingroup CPP_API_INTERNAL_GROUP
	 */
	class DynLib
	{
		public:
			/**
			 * @brief Loads a shared object by its filename
			 * @param name Filename
			 * @param silentfail If false gives some information to stdout
			 */
			DynLib(const std::string& name, bool silentfail = false);
			~DynLib();
			/**
			 * @brief Loads a function from the shared object
			 * @param name Function name to load
			 * @param silentfail If false gives some information to stdout
			 * @return
			 */
			void* loadFunction(const char* name, bool silentfail = false);
			/**
			 * @brief Check if the shared object was loaded successfully
			 * @return Success
			 */
			bool isValid() const;
			/**
			 * @brief Returns the platforms shared object suffix
			 * @return Suffix (.dll/.so/...)
			 */
			static std::string getSuffix();
		private:
			DynLibHandle handle;
	};
}
#endif
