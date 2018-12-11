BIND_CLASS( Population
  CONSTRUCTOR NAME "createData" ARGS Host& host int size
  DESTRUCTOR NAME "destroyData"
  FUNCTION getOrbit RETURN Orbit*
  FUNCTION getObjectProperties RETURN ObjectProperties*
  FUNCTION getPosition RETURN Vector3*
  FUNCTION getVelocity RETURN Vector3*
  FUNCTION getAcceleration RETURN Vector3*
  FUNCTION getCovariance RETURN Covariance*
  FUNCTION getSize RETURN int
  FUNCTION getBytes RETURN Char*
  FUNCTION resizeByteArray ARGS int size
  FUNCTION update RETURN ErrorCode ARGS int type
)

