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
#include "opi_propagator.h"
#include "opi_host.h"
#include "opi_perturbation_module.h"
#include "opi_indexlist.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <iomanip>

namespace OPI
{
	/**
	 * \cond INTERNAL_DOCUMENTATION
	 */
	class PropagatorImpl
	{
		public:
			PropagatorImpl():
				allowPerturbationModules(false)
			{
			}

			bool allowPerturbationModules;
			std::vector<PerturbationModule*> perturbationModules;
	};

	//! \endcond

    Propagator::Propagator()
	{
        configFileName = "";
	}

	Propagator::~Propagator()
	{
	}

	void Propagator::useModules()
	{
		data->allowPerturbationModules = true;
	}

	bool Propagator::usesModules() const
	{
		return data->allowPerturbationModules;
	}

	PerturbationModule* Propagator::getPerturbationModule(int index)
	{
		return data->perturbationModules[index];
	}

	int Propagator::getPerturbationModuleCount() const
	{
		return data->perturbationModules.size();
	}

    ErrorCode Propagator::propagate(Population& population, double julian_day, double dt, PropagationMode mode, IndexList* indices)
	{
		ErrorCode status = SUCCESS;
		// ensure this propagator is enabled
		status = enable();
		// an error occured?
		if(status == SUCCESS)
            status = runPropagation(population, julian_day, dt, mode, indices);
		getHost()->sendError(status);
        if (status == SUCCESS && population.getLastPropagatorName() != getName())
        {
            population.setLastPropagatorName(getName());
        }
		return status;
	}

	bool Propagator::backwardPropagation()
	{
        return false;
	}

	bool Propagator::cartesianCoordinates()
	{
		return false;
	}

    ReferenceFrame Propagator::referenceFrame()
    {
        if (cartesianCoordinates())
        {
            return REF_UNSPECIFIED;
        }
        else return REF_NONE;
    }

    CovarianceType Propagator::covarianceType()
    {
        return CV_NONE;
    }

    ErrorCode Propagator::loadPopulation(Population& population, const char* filename)
    {
        return NOT_IMPLEMENTED;
    }

    void Propagator::loadConfigFile()
    {
        loadConfigFile(configFileName.c_str());
    }

    void Propagator::loadConfigFile(const char* filename)
    {
        std::string filenameStr(filename);
        if (filenameStr != "" && filenameStr.length() > 4)
        {
            if (filenameStr.substr(filenameStr.length()-4,4) == ".cfg")
            {
                std::ifstream in(filename, std::ifstream::in);
                if (in.is_open())
                {
                    std::cout << "Applying settings for " << getName() << " from config file" << std::endl;
                    while (in.good())
                    {
                        std::string line;
                        std::getline(in, line);
                        line = trim(line);
                        if (line[0] != '#')
                        {
                            std::vector<std::string> setting = tokenize(line, "=");
                            if (setting.size() >= 2)
                            {
                                std::string property = trim(setting[0]);
                                std::string value = trim(setting[1]);

                                if (value.substr(0,1) == "\"" && value.substr(value.length()-1, value.length()) == "\"")
                                {
                                    if (!hasProperty(property.c_str()))
                                    {
                                        std::cout << "Registering new PropagatorProperty from config file: " << property << " (string)" << std::endl;
                                        createProperty(property.c_str(), value.substr(1,value.length()-2).c_str());
                                    }
                                    else setProperty(property.c_str(), value.substr(1,value.length()-2).c_str());
                                }
                                else if (value.find_first_of(".") != std::string::npos)
                                {
                                    if (value.substr(value.length()-1,1) == "f")
                                    {
                                        if (!hasProperty(property.c_str()))
                                        {
                                            std::cout << "Registering new PropagatorProperty from config file: " << property << " (float)" << std::endl;
                                            createProperty(property.c_str(), (float)atof(value.substr(0,value.length()-2).c_str()));
                                        }
                                        else setProperty(property.c_str(), (float)atof(value.substr(0,value.length()-2).c_str()));
                                    }
                                    else {
                                        if (!hasProperty(property.c_str()))
                                        {
                                            std::cout << "Registering new PropagatorProperty from config file: " << property << " (double)" << std::endl;
                                            createProperty(property.c_str(), atof(value.c_str()));
                                        }
                                        else setProperty(property.c_str(), atof(value.c_str()));
                                    }
                                }
                                else {
                                    if (!hasProperty(property.c_str()))
                                    {
                                        std::cout << "Registering new PropagatorProperty from config file: " << property << " (int)" << std::endl;
                                        createProperty(property.c_str(), atoi(value.c_str()));
                                    }
                                    else setProperty(property.c_str(), atoi(value.c_str()));
                                }
                            }
                        }
                    }
                    in.close();
                }
                else {
                    //std::cout << "No config file found for propagator " << getName() << std::endl;
                }
            }
            else {
                std::cout << filenameStr << " is not a valid config file for propagator " << getName() << std::endl;
            }
        }
        configFileName = filenameStr;
    }

    OPI::ErrorCode Propagator::align(OPI::Population& population, double dt)
    {
        // Find the target epoch to align the population to
        double latestEpoch = 0.0;
        for (int i=0; i<population.getSize(); i++)
        {
            double currentEpoch = population.getEpoch()[i].current_epoch;
            if (currentEpoch == 0.0)
            {
                std::cout << "Cannot align: Current epoch must be set for all objects." << std::endl;
                return OPI::INVALID_DATA;
            }
            latestEpoch = std::max(latestEpoch, currentEpoch);
        }
        std::cout << "Aligning to epoch " << std::setprecision(15) << latestEpoch << std::endl;

        OPI::ErrorCode error;
        int objectsAligned = 0;
        while (objectsAligned < population.getSize())
        {
            objectsAligned = 0;
            OPI::IndexList trailingObjects(population.getHostPointer());
            for (int i=0; i<population.getSize(); i++)
            {                
                const double currentEpoch = population.getEpoch()[i].current_epoch;
                const double deltaSeconds = (latestEpoch - currentEpoch) * 86400.0;
                if (deltaSeconds < 1)
                {
                    objectsAligned++;
                }
                else if (deltaSeconds >= dt)
                {
                    trailingObjects.add(i);
                }
                else if (deltaSeconds < dt)
                {
                    // Object is close to the target epoch. Propagate individually.
                    OPI::IndexList thisObject(population.getHostPointer());
                    thisObject.add(i);
                    //std::cout << "Object " << i << " closing in. Propagating for " << deltaSeconds << " seconds." << std::endl;
                    error = runPropagation(population, 0.0, deltaSeconds, OPI::MODE_INDIVIDUAL_EPOCHS, &thisObject);
                }
            }
            if (trailingObjects.getSize() > 0) error = runPropagation(population, 0.0, dt, OPI::MODE_INDIVIDUAL_EPOCHS, &trailingObjects);
            if (error == OPI::NOT_IMPLEMENTED)
            {
                std::cout << "Cannot align: Propagator does not support the required functions." << std::endl;
                return error;
            }
            else if (error != OPI::SUCCESS)
            {
                std::cout << "Propagator returned an error during object alignment." << std::endl;
                return error;
            }
            //std::cout << objectsAligned << " objects aligned." << std::endl;
        }
        //std::cout << "Alignment complete." << std::endl;
        return OPI::SUCCESS;
    }

    std::vector<std::string> Propagator::tokenize(std::string line, std::string delimiter)
    {
        std::vector<std::string> elements;

        std::string::size_type lastPos = line.find_first_not_of(delimiter, 0);
        std::string::size_type pos     = line.find_first_of(delimiter, lastPos);

        while (std::string::npos != pos || std::string::npos != lastPos)
        {
            elements.push_back(line.substr(lastPos, pos - lastPos));
            lastPos = line.find_first_not_of(delimiter, pos);
            pos = line.find_first_of(delimiter, lastPos);
        }
        return elements;
    }

    std::string Propagator::trim(const std::string &s)
    {
        auto wsfront = std::find_if_not(s.begin(), s.end(), [](int c){return isspace(c); });
        auto wsback = std::find_if_not(s.rbegin(), s.rend(), [](int c){return isspace(c); }).base();
        return (wsback <= wsfront ? std::string() : std::string(wsfront, wsback));
    }

}
