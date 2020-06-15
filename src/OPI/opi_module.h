#ifndef OPI_MODULE_H
#define OPI_MODULE_H
#include "opi_common.h"
#include "opi_population.h"
#include "opi_error.h"
#include "opi_pimpl_helper.h"
#include <string>
#include <vector>

namespace OPI
{
	class Population;
	class IndexList;

	//! Contains the module implementation data
	class ModuleImpl;

	//! \brief This interface class defines the common module shared functions
	//! \ingroup CPP_API_GROUP
	class Module
	{
		public:
			OPI_API_EXPORT Module();
			OPI_API_EXPORT virtual ~Module();

			//! Checks if this propagator is enabled
			OPI_API_EXPORT bool isEnabled() const;
			//! Enable this propagator (reserves internal memory)
			OPI_API_EXPORT ErrorCode enable();
			//! Disable this propagator (frees up internal memory)
			OPI_API_EXPORT ErrorCode disable();
			//! Sets the name of this module
            OPI_API_EXPORT void setName(const char* name);
			//! Returns the name of this module
            OPI_API_EXPORT const char* getName() const;
			//! Sets the author of this module
            OPI_API_EXPORT void setAuthor(const char* name);
			//! Returns the author of this module
            OPI_API_EXPORT const char* getAuthor() const;
			//! Sets the description of this module
            OPI_API_EXPORT void setDescription(const char* name);
			//! Returns the description of this module
            OPI_API_EXPORT const char* getDescription() const;

            /**
             * @brief loadConfigFile Attempts to load the standard configuration file.
             *
             * The standard configuration file resides in the same directory as the plugin
             * with a .cfg suffix instead of the platform's library extension (.dll, .so,
             * .dynlib). This file is automatically loaded by the host on initialization using
             * the second variant of this function below.
             * It is recommended for plugin authors to call this function on runDisable()
             * as part of resetting the propagator to its default state.
             */
            OPI_API_EXPORT void loadConfigFile();

            /**
             * @brief loadConfigFile Attempts to load a config file. Called by the host upon initialization.
             *
             * This function will automatically be called by OPI (and, in most cases, should
             * only ever be called by OPI) when the plugin is first loaded.
             * The given config file name will be stored in the propagator. Plugin authors
             * should use the above variant of this function when resetting the propagator.
             * @param filename The name of the config file to load.
             */
            OPI_API_EXPORT void loadConfigFile(const char* filename);

            /**
             * @brief loadResource Load the given resource from the standard resource archive.
             *
             * The resource archive is a zip archive with the same base name as the plugin (and
             * config file) and a ".dat" suffix. Propagator authors can use it to store files that
             * are required for operation and access the resources inside by using this function.
             * @param resname The path of the file inside the resource archive.
             * @param buffer A buffer to hold the resource's contents. Must be freed by the caller
             * if returned size is larger than zero.
             * @return The size in bytes of the loaded resource.
             */
            OPI_API_EXPORT size_t loadResource(const char* resname, const char** buffer);

            /* NOT YET IMPLEMENTED
			//! Sets the version number of this module
			void setVersion(int major, int minor, int patch);
			//! Gets the major version of this module
			int getVersionMajor() const;
			//! Gets the minor version of this module
			int getVersionMinor() const;
			//! Gets the patch version of this module
			int getVersionPatch() const;
            */

			// property access functions
			//! registers a property
            OPI_API_EXPORT void registerProperty(const char* name, int* location);
			//! registers a property
            OPI_API_EXPORT void registerProperty(const char* name, float* location);
			//! registers a property
            OPI_API_EXPORT void registerProperty(const char* name, double* location);
			//! registers a property
            OPI_API_EXPORT void registerProperty(const char* name, int* location, int size);
			//! registers a property
            OPI_API_EXPORT void registerProperty(const char* name, float* location, int size);
			//! registers a property
            OPI_API_EXPORT void registerProperty(const char* name, double* location, int size);
			//! registers a property
            OPI_API_EXPORT void registerProperty(const char* name, std::string* location);

			//! creates a new property of type int, the memory will be managed by OPI
            OPI_API_EXPORT void createProperty(const char* name, int value);
			//! creates a new property of type float, the memory will be managed by OPI
            OPI_API_EXPORT void createProperty(const char* name, float value);
			//! creates a new property of type double, the memory will be managed by OPI
            OPI_API_EXPORT void createProperty(const char* name, double value);
			//! creates a new property of type string, the memory will be managed by OPI
            OPI_API_EXPORT void createProperty(const char* name, const char* value);

            /* NOT YET IMPLEMENTED
			//! creates a new property array of type int, the memory will be managed by OPI
            void createProperty(const char* name, int* value, int size);
			//! creates a new property array of type float, the memory will be managed by OPI
            void createProperty(const char* name, float* value, int size);
			//! creates a new property array of type double, the memory will be managed by OPI
            void createProperty(const char* name, double* value, int size);
            */

			//! Sets a property
            OPI_API_EXPORT ErrorCode setProperty(const char* name, int value);
			//! Sets a property
            OPI_API_EXPORT ErrorCode setProperty(const char* name, float value);
			//! Sets a property
            OPI_API_EXPORT ErrorCode setProperty(const char* name, double value);
			//! Sets a property
            OPI_API_EXPORT ErrorCode setProperty(const char* name, const char* value);
			//! Sets a property
            OPI_API_EXPORT ErrorCode setProperty(const char* name, int* value, int n);
			//! Sets a property
            OPI_API_EXPORT ErrorCode setProperty(const char* name, float* value, int n);
			//! Sets a property
            OPI_API_EXPORT ErrorCode setProperty(const char* name, double* value, int n);
			//! Gets the value of a given property
            OPI_API_EXPORT int getPropertyInt(const char* name, int element = 0);
			//! Gets the value of a given property
			OPI_API_EXPORT int getPropertyInt(int index, int element = 0);
			//! Gets the value of a given property
            OPI_API_EXPORT float getPropertyFloat(const char* name, int element = 0);
			//! Gets the value of a given property
			OPI_API_EXPORT float getPropertyFloat(int index, int element = 0);
			//! Gets the value of a given property
            OPI_API_EXPORT double getPropertyDouble(const char* name, int element = 0);
			//! Gets the value of a given property
			OPI_API_EXPORT double getPropertyDouble(int index, int element = 0);
			//! Gets the value of a given property
            OPI_API_EXPORT const char* getPropertyString(const char* name, int element = 0);
			//! Gets the value of a given property
            OPI_API_EXPORT const char* getPropertyString(int index, int element = 0);

			// property information functions
			//! Returns the amount of registered properties
			OPI_API_EXPORT int getPropertyCount() const;
			//! Returns the name of the property identified by the given index
            OPI_API_EXPORT const char* getPropertyName(int index) const;
			//! Checks if a property is registered
            OPI_API_EXPORT bool hasProperty(const char* name) const;
			//! Returns the type of a property
            OPI_API_EXPORT PropertyType getPropertyType(const char* name) const;
			//! Returns the type of the property identified by the given index
			OPI_API_EXPORT PropertyType getPropertyType(int index) const;
			//! Returns the size of the property
            OPI_API_EXPORT int getPropertySize(const char* name) const;
			//! Returns the size of the property
			OPI_API_EXPORT int getPropertySize(int index) const;


			//! Sets a private module-internal data pointer
			OPI_API_EXPORT void setPrivateData(void* private_data);
			//! Returns the module-internal data pointer
			OPI_API_EXPORT void* getPrivateData();

            /**
             * @brief requiresCUDA Check whether this module requires CUDA to function.
             * @return 0 if CUDA is not required, or the major number of
             *  the minimum required compute capability.
             */
            virtual int requiresCUDA();

            /**
             * @brief Check whether this module requires OpenCL to function.
             * @return 0 if OpenCL is not required, otherwise set this to the major number of
             * the required OpenCL version.
             */
            virtual int requiresOpenCL();

            /**
             * @brief minimumOPIVersionRequired Returns the minimum OPI API level that this module requires.
             *
             * API level is equal to OPI's major version number.
             * @return An integer representing the minimum API level required.
             */
            virtual int minimumOPIVersionRequired();

            /**
             * @brief supportsOPILogger Check whether this modules uses the OPI Logger.
             *
             * @return 0 if this module does not use OPI::Logger for message output (default),
             * 1 otherwise.
             */
            virtual int supportsOPILogger();

		protected:
			//! Returns the Host of this propagator
			Host* getHost() const;
			//! Override this if you want to change the enable behaviour
			virtual ErrorCode runEnable();
			//! Override this if you want to change the disable behaviour
			virtual ErrorCode runDisable();

		private:
			//! \cond INTERNAL_DOCUMENTATION
			void setHost(OPI::Host* newhost);
			friend class Host;
			//! \endcond
			Pimpl<ModuleImpl> data;

            //! Auxiliary functions for loadConfig
            std::vector<std::string> tokenize(std::string line, std::string delimiter);
            std::string trim(const std::string &s);
	};
}

#endif
