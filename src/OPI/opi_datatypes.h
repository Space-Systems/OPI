#ifndef OPI_DATA_TYPES_H
#define OPI_DATA_TYPES_H

#ifndef OPI_CUDA_PREFIX
#define OPI_CUDA_PREFIX
#endif

#include "OPI/opi_types.h"
#define _USE_MATH_DEFINES
#include <cmath>
#include <limits>
#include <string>

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

    //! Addition assignment operator for Orbit
    OPI_CUDA_PREFIX inline Orbit& operator+=(Orbit& a, const Orbit& b)
    {
        a.semi_major_axis += b.semi_major_axis;
        a.eccentricity += b.eccentricity;
        a.inclination += b.inclination;
        a.raan += b.raan;
        a.arg_of_perigee += b.arg_of_perigee;
        a.mean_anomaly += b.mean_anomaly;
        return a;
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

    //! Calculates the square distance between two vectors
    OPI_CUDA_PREFIX inline double distanceSquared(const Vector3& a, const Vector3& b)
    {
        return (pow(a.x-b.x,2) + pow(a.y-b.y,2) + pow(a.z-b.z,2));
    }

    //! Calculates the distance between two vectors
    OPI_CUDA_PREFIX inline double distance(const Vector3& a, const Vector3& b)
    {
        return sqrt(distanceSquared(a,b));
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

    OPI_CUDA_PREFIX inline bool isZero(const Orbit& o)
    {
        return (o.semi_major_axis == 0.0
                && o.eccentricity == 0.0
                && o.inclination == 0.0
                && o.raan == 0.0
                && o.arg_of_perigee == 0.0
                && o.mean_anomaly == 0.0);
    }

    OPI_CUDA_PREFIX inline bool isZero(const Vector3& v)
    {
        return (v.x == 0.0 && v.y == 0.0 && v.z == 0.0);
    }

    OPI_CUDA_PREFIX inline bool isZero(const Epoch& e)
    {
        return (e.beginning_of_life == 0.0 && e.end_of_life == 0.0 && e.current_epoch == 0.0);
    }

    OPI_CUDA_PREFIX inline bool isZero(const ObjectProperties& p)
    {
        // ID can be zero
        return (p.mass == 0.0
                && p.diameter == 0.0
                && p.area_to_mass == 0.0
                && p.drag_coefficient == 0.0
                && p.reflectivity == 0.0
                );
    }

    OPI_CUDA_PREFIX inline bool isZero(const Covariance& c)
    {
        return (c.k1_k1 == 0.0 &&
                c.k2_k1 == 0.0 &&
                c.k2_k2 == 0.0 &&
                c.k3_k1 == 0.0 &&
                c.k3_k2 == 0.0 &&
                c.k3_k3 == 0.0 &&
                c.k4_k1 == 0.0 &&
                c.k4_k2 == 0.0 &&
                c.k4_k3 == 0.0 &&
                c.k4_k4 == 0.0 &&
                c.k5_k1 == 0.0 &&
                c.k5_k2 == 0.0 &&
                c.k5_k3 == 0.0 &&
                c.k5_k4 == 0.0 &&
                c.k5_k5 == 0.0 &&
                c.k6_k1 == 0.0 &&
                c.k6_k2 == 0.0 &&
                c.k6_k3 == 0.0 &&
                c.k6_k4 == 0.0 &&
                c.k6_k5 == 0.0 &&
                c.k6_k6 == 0.0 &&
                c.d1_k1 == 0.0 &&
                c.d1_k2 == 0.0 &&
                c.d1_k3 == 0.0 &&
                c.d1_k4 == 0.0 &&
                c.d1_k5 == 0.0 &&
                c.d1_k6 == 0.0 &&
                c.d1_d1 == 0.0 &&
                c.d2_k1 == 0.0 &&
                c.d2_k2 == 0.0 &&
                c.d2_k3 == 0.0 &&
                c.d2_k4 == 0.0 &&
                c.d2_k5 == 0.0 &&
                c.d2_k6 == 0.0 &&
                c.d2_d1 == 0.0 &&
                c.d2_d2 == 0.0
                );
    }


    OPI_CUDA_PREFIX inline bool hasNaN(const Orbit& o)
    {
        return (std::isnan(o.semi_major_axis)
                || std::isnan(o.eccentricity)
                || std::isnan(o.inclination)
                || std::isnan(o.raan)
                || std::isnan(o.arg_of_perigee)
                || std::isnan(o.mean_anomaly));
    }

    OPI_CUDA_PREFIX inline bool hasNaN(const Vector3& v)
    {
        return (std::isnan(v.x) || std::isnan(v.y) || std::isnan(v.z));
    }

    OPI_CUDA_PREFIX inline bool hasNaN(const ObjectProperties& p)
    {
        return (std::isnan(p.mass)
                || std::isnan(p.diameter)
                || std::isnan(p.area_to_mass)
                || std::isnan(p.drag_coefficient)
                || std::isnan(p.reflectivity));
    }

    //! Addition assignment operator for VMatrix
    OPI_CUDA_PREFIX inline PartialsMatrix& operator+=(PartialsMatrix& a, const PartialsMatrix& b)
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

        a.accX_k1 += b.accX_k1;
        a.accY_k1 += b.accY_k1;
        a.accZ_k1 += b.accZ_k1;
        a.accX_k2 += b.accX_k2;
        a.accY_k2 += b.accY_k2;
        a.accZ_k2 += b.accZ_k2;
        a.accX_k3 += b.accX_k3;
        a.accY_k3 += b.accY_k3;
        a.accZ_k3 += b.accZ_k3;

        a.accX_k4 += b.accX_k4;
        a.accY_k4 += b.accY_k4;
        a.accZ_k4 += b.accZ_k4;
        a.accX_k5 += b.accX_k5;
        a.accY_k5 += b.accY_k5;
        a.accZ_k5 += b.accZ_k5;
        a.accX_k6 += b.accX_k6;
        a.accY_k6 += b.accY_k6;
        a.accZ_k6 += b.accZ_k6;

        return a;
    }

    //! writes the PartialsMatrix to a C vector (no bounds check!)
    OPI_CUDA_PREFIX inline void partialsToArray(const PartialsMatrix m, double* v)
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

        v[18] = m.accX_k1;
        v[19] = m.accY_k1;
        v[20] = m.accZ_k1;

        v[21] = m.accX_k2;
        v[22] = m.accY_k2;
        v[23] = m.accZ_k2;

        v[24] = m.accX_k3;
        v[25] = m.accY_k3;
        v[26] = m.accZ_k3;

        v[27] = m.accX_k4;
        v[28] = m.accY_k4;
        v[29] = m.accZ_k4;

        v[30] = m.accX_k5;
        v[31] = m.accY_k5;
        v[32] = m.accZ_k5;

        v[33] = m.accX_k6;
        v[34] = m.accY_k6;
        v[35] = m.accZ_k6;
    }

    //! Sets the PartialsMatrix from a C vector (no bounds check!)
    OPI_CUDA_PREFIX inline PartialsMatrix arrayToPartials(double* v)
    {
        PartialsMatrix m;
        m.accX_posX = v[0];
        m.accY_posX = v[1];
        m.accZ_posX = v[2];
        m.accX_posY = v[3];
        m.accY_posY = v[4];
        m.accZ_posY = v[5];
        m.accX_posZ = v[6];
        m.accY_posZ = v[7];
        m.accZ_posZ = v[8];

        m.accX_velX = v[9];
        m.accY_velX = v[10];
        m.accZ_velX = v[11];
        m.accX_velY = v[12];
        m.accY_velY = v[13];
        m.accZ_velY = v[14];
        m.accX_velZ = v[15];
        m.accY_velZ = v[16];
        m.accZ_velZ = v[17];

        m.accX_k1 = v[18];
        m.accY_k1 = v[19];
        m.accZ_k1 = v[20];

        m.accX_k2 = v[21];
        m.accY_k2 = v[22];
        m.accZ_k2 = v[23];

        m.accX_k3 = v[24];
        m.accY_k3 = v[25];
        m.accZ_k3 = v[26];

        m.accX_k4 = v[27];
        m.accY_k4 = v[28];
        m.accZ_k4 = v[29];

        m.accX_k5 = v[30];
        m.accY_k5 = v[31];
        m.accZ_k5 = v[32];

        m.accX_k6 = v[33];
        m.accY_k6 = v[34];
        m.accZ_k6 = v[35];

        return m;
    }


    //! writes the covariance matrix (lower triangular) to a C vector (no bounds check!)
    OPI_CUDA_PREFIX inline void covarianceToArray(const Covariance c, double* v)
    {
        v[0] = c.k1_k1;

        v[1] = c.k2_k1;
        v[2] = c.k2_k2;

        v[3] = c.k3_k1;
        v[4] = c.k3_k2;
        v[5] = c.k3_k3;

        v[6] = c.k4_k1;
        v[7] = c.k4_k2;
        v[8] = c.k4_k3;
        v[9] = c.k4_k4;

        v[10] = c.k5_k1;
        v[11] = c.k5_k2;
        v[12] = c.k5_k3;
        v[13] = c.k5_k4;
        v[14] = c.k5_k5;

        v[15] = c.k6_k1;
        v[16] = c.k6_k2;
        v[17] = c.k6_k3;
        v[18] = c.k6_k4;
        v[19] = c.k6_k5;
        v[20] = c.k6_k6;

        v[21] = c.d1_k1;
        v[22] = c.d1_k2;
        v[23] = c.d1_k3;
        v[24] = c.d1_k4;
        v[25] = c.d1_k5;
        v[26] = c.d1_k6;
        v[27] = c.d1_d1;

        v[28] = c.d2_k1;
        v[29] = c.d2_k2;
        v[30] = c.d2_k3;
        v[31] = c.d2_k4;
        v[32] = c.d2_k5;
        v[33] = c.d2_k6;
        v[34] = c.d2_d1;
        v[35] = c.d2_d2;
    }

    OPI_CUDA_PREFIX inline Covariance arrayToCovariance(double* v)
    {
        Covariance c;
        c.k1_k1 = v[0];

        c.k2_k1 = v[1];
        c.k2_k2 = v[2];

        c.k3_k1 = v[3];
        c.k3_k2 = v[4];
        c.k3_k3 = v[5];

        c.k4_k1 = v[6];
        c.k4_k2 = v[7];
        c.k4_k3 = v[8];
        c.k4_k4 = v[9];

        c.k5_k1 = v[10];
        c.k5_k2 = v[11];
        c.k5_k3 = v[12];
        c.k5_k4 = v[13];
        c.k5_k5 = v[14];

        c.k6_k1 = v[15];
        c.k6_k2 = v[16];
        c.k6_k3 = v[17];
        c.k6_k4 = v[18];
        c.k6_k5 = v[19];
        c.k6_k6 = v[20];

        c.d1_k1 = v[21];
        c.d1_k2 = v[22];
        c.d1_k3 = v[23];
        c.d1_k4 = v[24];
        c.d1_k5 = v[25];
        c.d1_k6 = v[26];
        c.d1_d1 = v[27];

        c.d2_k1 = v[28];
        c.d2_k2 = v[29];
        c.d2_k3 = v[30];
        c.d2_k4 = v[31];
        c.d2_k5 = v[32];
        c.d2_k6 = v[33];
        c.d2_d1 = v[34];
        c.d2_d2 = v[35];

        return c;
    }

    inline const char* referenceFrameToString(ReferenceFrame rf)
    {
        switch (rf)
        {
        case REF_NONE: return "NONE";
        case REF_UNSPECIFIED: return "UNSPECIFIED";
        case REF_TEME: return "TEME";
        case REF_GCRF: return "GCRF";
        case REF_ITRF: return "ITRF";
        case REF_ECI: return "ECI";
        case REF_ECEF: return "ECEF";
        case REF_MOD: return "MOD";
        case REF_TOD: return "TOD";
        case REF_TOR: return "TOR";
        case REF_J2000: return "J2000";
        case REF_MULTIPLE: return "MULTIPLE";
        case REF_UNLISTED: return "UNLISTED";
        }
        return "";
    }

    inline ReferenceFrame referenceFrameFromString(const char* frame)
    {
        std::string f(frame);
        if (f == "NONE") return REF_NONE;
        else if (f == "UNSPECIFIED") return REF_UNSPECIFIED;
        else if (f == "TEME") return REF_TEME;
        else if (f == "GCRF") return REF_GCRF;
        else if (f == "ITRF") return REF_ITRF;
        else if (f == "ITRF") return REF_ITRF;
        else if (f == "ECI") return REF_ECI;
        else if (f == "ECEF") return REF_ECEF;
        else if (f == "MOD") return REF_MOD;
        else if (f == "TOD") return REF_TOD;
        else if (f == "TOR") return REF_TOR;
        else if (f == "J2000") return REF_J2000;
        else if (f == "MULTIPLE") return REF_MULTIPLE;
        else if (f == "UNLISTED") return REF_UNLISTED;

        return REF_UNSPECIFIED;
    }

}

#endif
