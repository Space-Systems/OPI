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
#include <limits>

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
    OPI_CUDA_PREFIX inline Orbit operator*(const Orbit& a, const double& b)
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
    OPI_CUDA_PREFIX inline Orbit operator/(const Orbit& a, const double& b)
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

    //! Calculate eccentric anomaly from given orbit
    OPI_CUDA_PREFIX inline double eccentricAnomaly(const Orbit& o)
    {
        double ea = o.mean_anomaly;
        int maxloop = 5;
        double fcte, fctes;

        for (int i=0; i<maxloop; i++) {
            fcte  = ea - o.eccentricity * sin(ea) - o.mean_anomaly;
            fctes = 1.0 - o.eccentricity * cos(ea);
            ea -= fcte/fctes;
        }
        return ea;
    }

    //! Calculate true anomaly from given orbit
    OPI_CUDA_PREFIX inline double trueAnomaly(const Orbit& o)
    {
        double ea = eccentricAnomaly(o);
        const double sin_ea = sin(ea/2.0);
        double cos_ea = cos(ea/2.0);
        const double epsilon = 1e-15;

        if (fabs(cos_ea) < epsilon) cos_ea = epsilon * (cos_ea < 0.0 ? -1.0 : 1.0);

        double ta = 2.0 * atan(
            sqrt((1.0 + o.eccentricity)/(1.0 - o.eccentricity)) * sin_ea/cos_ea
        );
        return ta;
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

    //! Addition assignment operator for Vector3
    OPI_CUDA_PREFIX inline Vector3& operator+=(Vector3& a, const Vector3& b)
    {
        a.x += b.x;
        a.y += b.y;
        a.z += b.z;
        return a;
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

    OPI_CUDA_PREFIX inline Vector3 operator+(const Vector3& a, double b)
	{
		Vector3 out;
		out.x = a.x + b;
		out.y = a.y + b;
		out.z = a.z + b;
		return out;
	}

    OPI_CUDA_PREFIX inline Vector3 operator-(const Vector3& a, double b)
	{
		Vector3 out;
		out.x = a.x - b;
		out.y = a.y - b;
		out.z = a.z - b;
		return out;
	}

    //! Divides a vector by a double
    OPI_CUDA_PREFIX inline Vector3 operator/(const Vector3& a, double b)
	{
		Vector3 out;
		out.x = a.x / b;
		out.y = a.y / b;
		out.z = a.z / b;
		return out;
	}

    //! Multiplies a vector times a double
    OPI_CUDA_PREFIX inline Vector3 operator*(const Vector3& a, double b)
	{
		Vector3 out;
		out.x = a.x * b;
		out.y = a.y * b;
		out.z = a.z * b;
		return out;
	}

	//! Dot product for vector3
    OPI_CUDA_PREFIX inline double operator*(const Vector3& a, const Vector3& b)
	{
        double out;
		out = a.x * b.x + a.y * b.y + a.z * b.z;
		return out;
	}

	OPI_CUDA_PREFIX inline Vector3 cross(const Vector3& a, const Vector3& b)
	{
		Vector3 out;
		out.x = a.y * b.z - a.z * b.y;
		out.y = a.z * b.x - a.x * b.z;
		out.z = a.x * b.y - a.y * b.x;
		return out;
	}

	//! Calculates the length²
    OPI_CUDA_PREFIX inline double lengthSquare(const Vector3& v)
	{
		return v.x * v.x + v.y * v.y + v.z * v.z;
	}

	//! Calculates the length
    OPI_CUDA_PREFIX inline double length(const Vector3 v)
	{
		return sqrt(lengthSquare(v));
	}

	//! Returns the smallest element
    OPI_CUDA_PREFIX inline double smallest(const Vector3& v)
	{
		if(v.x < v.y)
			return v.x < v.z ? v.x : v.z;
		else
			return v.y < v.z ? v.y : v.z;
	}

	//! Returns the highest element
    OPI_CUDA_PREFIX inline double highest(const Vector3& v)
	{
		if(v.x > v.y)
			return v.x > v.z ? v.x : v.z;
		else
			return v.y > v.z ? v.y : v.z;
	}

    //! Returns the given vector, rotate by angle degrees around the X axis
    OPI_CUDA_PREFIX inline Vector3 rotateX(const Vector3& v, const double angle)
    {
        Vector3 r;
        r.x = v.x;
        r.y = cos(angle) * v.y + sin(angle) * v.z;
        r.z = cos(angle) * v.z - sin(angle) * v.y;
        return r;
    }

    //! Returns the given vector, rotate by angle degrees around the Y axis
    OPI_CUDA_PREFIX inline Vector3 rotateY(const Vector3& v, const double angle)
    {
        Vector3 r;
        r.x = cos(angle) * v.x - sin(angle) * v.z;
        r.y = v.y;
        r.z = cos(angle) * v.z + sin(angle) * v.x;
        return r;
    }

    //! Returns the given vector, rotate by angle degrees around the Z axis
    OPI_CUDA_PREFIX inline Vector3 rotateZ(const Vector3& v, const double angle)
    {
        Vector3 r;
        r.x = cos(angle) * v.x + sin(angle) * v.y;
        r.y = cos(angle) * v.y - sin(angle) * v.x;
        r.z = v.z;
        return r;
    }

    //! Returns the angle, in radians, between two given vectors, nan if undefined
    OPI_CUDA_PREFIX inline double angle(const Vector3& a, const Vector3& b)
    {
        const double magnitude = length(a) * length(b);
        const double epsilon = 1e-15;
        if (magnitude > pow(epsilon,2.0))
        {
            double cosAngle = (a*b) / magnitude;
            if (cosAngle > 1.0) cosAngle = 1.0;
            else if (cosAngle < -1.0) cosAngle = -1.0;
            return acos(cosAngle);
        }
        else return std::numeric_limits<double>::quiet_NaN();
    }

    //! Addition assignment operator for VMatrix
    OPI_CUDA_PREFIX inline VMatrix& operator+=(VMatrix& a, const VMatrix& b)
    {
        a.accX_posX += b.accX_posX;
        a.accY_posX += b.accY_posX;
        a.accZ_posX += b.accZ_posX;
        a.accX_posY += b.accX_posY;
        a.accY_posY += b.accY_posY;
        a.accZ_posY += b.accZ_posY;
        a.accX_posZ += b.accX_posZ;
        a.accY_posZ += b.accY_posZ;
        a.accZ_posZ += b.accZ_posZ;

        a.accX_velX += b.accX_velX;
        a.accY_velX += b.accY_velX;
        a.accZ_velX += b.accZ_velX;
        a.accX_velY += b.accX_velY;
        a.accY_velY += b.accY_velY;
        a.accZ_velY += b.accZ_velY;
        a.accX_velZ += b.accX_velZ;
        a.accY_velZ += b.accY_velZ;
        a.accZ_velZ += b.accZ_velZ;
        return a;
    }

    //! writes the VMatrix to a C vector (no bounds check!)
    OPI_CUDA_PREFIX inline void vMatrixToArray(const VMatrix m, double* v)
    {
        v[0] = m.accX_posX;
        v[1] = m.accY_posX;
        v[2] = m.accZ_posX;
        v[3] = m.accX_posY;
        v[4] = m.accY_posY;
        v[5] = m.accZ_posY;
        v[6] = m.accX_posZ;
        v[7] = m.accY_posZ;
        v[8] = m.accZ_posZ;

        v[9] = m.accX_velX;
        v[10] = m.accY_velX;
        v[11] = m.accZ_velX;
        v[12] = m.accX_velY;
        v[13] = m.accY_velY;
        v[14] = m.accZ_velY;
        v[15] = m.accX_velZ;
        v[16] = m.accY_velZ;
        v[17] = m.accZ_velZ;
    }

}

#endif
