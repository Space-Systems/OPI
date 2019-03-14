BIND_CLASS( Population
  CONSTRUCTOR NAME "createData" ARGS Host& host int size
  DESTRUCTOR NAME "destroyData"
  FUNCTION getOrbit RETURN Orbit*
  FUNCTION getObjectProperties RETURN ObjectProperties*
  FUNCTION getPosition RETURN Vector3*
  FUNCTION getVelocity RETURN Vector3*
  FUNCTION getAcceleration RETURN Vector3*
  FUNCTION getEpoch RETURN Epoch*
  FUNCTION getSize RETURN int
  FUNCTION update RETURN ErrorCode ARGS int type
)

