#include "opi_opencl_functions.h"

namespace OPI
{

const std::string OPENCL_FUNCTIONS = R"(
// Equivalent of OPI::Vector3 in OpenCL
typedef struct OPI_Vector3 {
    double x;
    double y;
    double z;
} OPI_Vector3;

// Equivalent of OPI::JulianDay in OpenCL
typedef struct OPI_JulianDay {
    int day;
    long usec;
} OPI_JulianDay;

// Equivalent of OPI::Epoch in OpenCL
typedef struct OPI_Epoch {
    OPI_JulianDay beginning_of_life;
    OPI_JulianDay end_of_life;
    OPI_JulianDay current_epoch;
    OPI_JulianDay original_epoch;
    OPI_JulianDay initial_epoch;
} OPI_Epoch;

// Equivalent of OPI::Orbit in OpenCL
typedef struct OPI_Orbit {
    double semi_major_axis;
    double eccentricity;
    double inclination;
    double raan;
    double arg_of_perigee;
    double mean_anomaly;
} OPI_Orbit;

typedef struct OPI_ObjectProperties {
    double mass;
    double diameter;
    double area_to_mass;
    double drag_coefficient;
    double reflectivity;
    int id;
} OPI_ObjectProperties;

typedef enum OPI_PropagationMode
{
    MODE_SINGLE_EPOCH = 0,
    MODE_INDIVIDUAL_EPOCHS = 1,
} OPI_PropagationMode;

typedef struct OPI_Covariance
{
    double k1_k1;
    double k2_k1;
    double k2_k2;
    double k3_k1;
    double k3_k2;
    double k3_k3;
    double k4_k1;
    double k4_k2;
    double k4_k3;
    double k4_k4;
    double k5_k1;
    double k5_k2;
    double k5_k3;
    double k5_k4;
    double k5_k5;
    double k6_k1;
    double k6_k2;
    double k6_k3;
    double k6_k4;
    double k6_k5;
    double k6_k6;
    double d1_k1;
    double d1_k2;
    double d1_k3;
    double d1_k4;
    double d1_k5;
    double d1_k6;
    double d1_d1;
    double d2_k1;
    double d2_k2;
    double d2_k3;
    double d2_k4;
    double d2_k5;
    double d2_k6;
    double d2_d1;
    double d2_d2;
} OPI_Covariance;

typedef struct OPI_PartialsMatrix
{
    double accX_posX;
    double accY_posX;
    double accZ_posX;
    double accX_posY;
    double accY_posY;
    double accZ_posY;
    double accX_posZ;
    double accY_posZ;
    double accZ_posZ;
    double accX_velX;
    double accY_velX;
    double accZ_velX;
    double accX_velY;
    double accY_velY;
    double accZ_velY;
    double accX_velZ;
    double accY_velZ;
    double accZ_velZ;
    double accX_k1;
    double accY_k1;
    double accZ_k1;
    double accX_k2;
    double accY_k2;
    double accZ_k2;
    double accX_k3;
    double accY_k3;
    double accZ_k3;
    double accX_k4;
    double accY_k4;
    double accZ_k4;
    double accX_k5;
    double accY_k5;
    double accZ_k5;
    double accX_k6;
    double accY_k6;
    double accZ_k6;
} OPI_PartialsMatrix;

bool OPI_marked_as_deorbited(OPI_Epoch e)
{
    return (e.end_of_life.day > 0 && e.current_epoch.day > 0 && (e.end_of_life.day > e.current_epoch.day || (e.end_of_life.day == e.current_epoch.day && e.end_of_life.usec >= e.current_epoch.usec)));
}

double3 OPI_to_double3(OPI_Vector3 v)
{
    return (double3)(v.x, v.y, v.z);
}

OPI_JulianDay OPI_adjustJD(OPI_JulianDay a)
{
    const long USEC_PER_DAY = 86400000000;
    if (a.usec >= USEC_PER_DAY || a.usec < 0)
    {
        a.day += a.usec / USEC_PER_DAY;
        a.usec = a.usec % USEC_PER_DAY;
    }
    if (a.usec < 0)
    {
        a.day -= 1;
        a.usec += USEC_PER_DAY;
    }
    return a;
}

OPI_JulianDay OPI_addJD(const OPI_JulianDay jd, const long usec)
{
    OPI_JulianDay result = jd;
    result.usec += usec;
    return OPI_adjustJD(result);
}

double OPI_toDouble(const OPI_JulianDay jd)
{
    return jd.day + (double)(jd.usec / 86400000000.0);
}

long OPI_deltaUsec(const OPI_JulianDay a, const OPI_JulianDay b)
{
    if (a.day > b.day || (a.day == b.day && a.usec > b.usec))
    {
        int day = a.day - b.day;
        long usec = a.usec - b.usec;
        return abs(day * 86400000000l + usec);
    }
    else {
        int day = b.day - a.day;
        long usec = b.usec - a.usec;
        return abs(day * 86400000000l + usec);
    }
}
)";

std::string getOpenCLCode() { return std::string(OPENCL_FUNCTIONS); }

}
