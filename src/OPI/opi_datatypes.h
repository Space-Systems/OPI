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
#ifndef OPI_DATA_TYPES_H
#define OPI_DATA_TYPES_H

#ifndef OPI_CUDA_PREFIX
#define OPI_CUDA_PREFIX
#endif

#include "OPI/opi_types.h"
#include <cmath>
namespace OPI
{
	//! Addition operator for Orbit
	OPI_CUDA_PREFIX inline Orbit operator+(const Orbit& a, const Orbit& b)
	{
		Orbit out;
		out.semi_major_axis = a.semi_major_axis + b.semi_major_axis;
		out.eccentricity = a.eccentricity + b.eccentricity;
		out.inclination = a.inclination + b.inclination;
		out.raan = a.raan + b.raan;
		out.arg_of_perigee = a.arg_of_perigee + b.arg_of_perigee;
		out.mean_anomaly = a.mean_anomaly + b.mean_anomaly;
		return out;
	}

	//! Subtraction operator for Orbit
	OPI_CUDA_PREFIX inline Orbit operator-(const Orbit& a, const Orbit& b)
	{
		Orbit out;
		out.semi_major_axis = a.semi_major_axis - b.semi_major_axis;
		out.eccentricity = a.eccentricity - b.eccentricity;
		out.inclination = a.inclination - b.inclination;
		out.raan = a.raan - b.raan;
		out.arg_of_perigee = a.arg_of_perigee - b.arg_of_perigee;
		out.mean_anomaly = a.mean_anomaly - b.mean_anomaly;
		return out;
	}

	//! Scalar Multiplication operator for Orbit
	OPI_CUDA_PREFIX inline Orbit operator*(const Orbit& a, const float& b)
	{
		Orbit out;
		out.semi_major_axis = a.semi_major_axis * b;
		out.eccentricity = a.eccentricity * b;
		out.inclination = a.inclination * b;
		out.raan = a.raan * b;
		out.arg_of_perigee = a.arg_of_perigee * b;
		out.mean_anomaly = a.mean_anomaly * b;
		return out;
	}

	//! Scalar Division operator for Orbit
	OPI_CUDA_PREFIX inline Orbit operator/(const Orbit& a, const float& b)
	{
		Orbit out;
		out.semi_major_axis = a.semi_major_axis / b;
		out.eccentricity = a.eccentricity / b;
		out.inclination = a.inclination / b;
		out.raan = a.raan / b;
		out.arg_of_perigee = a.arg_of_perigee / b;
		out.mean_anomaly = a.mean_anomaly / b;
		return out;
	}

	//! Addition operator for Vector3
	OPI_CUDA_PREFIX inline Vector3 operator+(const Vector3& a, const Vector3& b)
	{
		Vector3 out;
		out.x = a.x + b.x;
		out.y = a.y + b.y;
		out.z = a.z + b.z;
		return out;
	}

	//! Subtraction operator for Vector3
	OPI_CUDA_PREFIX inline Vector3 operator-(const Vector3& a, const Vector3& b)
	{
		Vector3 out;
		out.x = a.x - b.x;
		out.y = a.y - b.y;
		out.z = a.z - b.z;
		return out;
	}

	OPI_CUDA_PREFIX inline Vector3 operator+(const Vector3& a, float b)
	{
		Vector3 out;
		out.x = a.x + b;
		out.y = a.y + b;
		out.z = a.z + b;
		return out;
	}

	OPI_CUDA_PREFIX inline Vector3 operator-(const Vector3& a, float b)
	{
		Vector3 out;
		out.x = a.x - b;
		out.y = a.y - b;
		out.z = a.z - b;
		return out;
	}

	//! Multiplies a vector times a float
	OPI_CUDA_PREFIX inline Vector3 operator*(const Vector3& a, float b)
	{
		Vector3 out;
		out.x = a.x * b;
		out.y = a.y * b;
		out.z = a.z * b;
		return out;
	}

	//! Calculetes the length²
	OPI_CUDA_PREFIX inline float lengthSquare(const Vector3& v)
	{
		return v.x * v.x + v.y * v.y + v.z * v.z;
	}

	//! Calculates the length
	OPI_CUDA_PREFIX inline float length(const Vector3 v)
	{
		return sqrt(lengthSquare(v));
	}

	//! Returns the smallest element
	OPI_CUDA_PREFIX inline float smallest(const Vector3& v)
	{
		if(v.x < v.y)
			return v.x < v.z ? v.x : v.z;
		else
			return v.y < v.z ? v.y : v.z;
	}

	//! Returns the highest element
	OPI_CUDA_PREFIX inline float highest(const Vector3& v)
	{
		if(v.x > v.y)
			return v.x > v.z ? v.x : v.z;
		else
			return v.y > v.z ? v.y : v.z;
	}
}

#endif
