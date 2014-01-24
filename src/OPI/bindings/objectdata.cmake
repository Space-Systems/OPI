BIND_CLASS( ObjectData
  CONSTRUCTOR NAME "createData" ARGS Host& host int size
  DESTRUCTOR NAME "destroyData"
  FUNCTION getOrbit RETURN Orbit*
  FUNCTION getSize RETURN int
  FUNCTION update RETURN ErrorCode ARGS int type
)
