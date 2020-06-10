#REGISTER_CALLBACK(ErrorCallback ARGS Host host int errorcode void* privatedata)
#BIND_FUNCTION( ErrorMessage ARGS ErrorCode code RETURN cchar* message_string)

DECLARE_CLASS( Propagator )
DECLARE_CLASS( DistanceQuery )



BIND_CLASS( Host
  PREFIX "OPI_"
  CONSTRUCTOR
  DESTRUCTOR
  FUNCTION loadPlugins RETURN ErrorCode ARGS "const char*" plugindir
  FUNCTION getLastError RETURN ErrorCode
  FUNCTION setErrorCallback ARGS ErrorCallback callback
                                 void* privatedata
  FUNCTION getPropagator OVERLOAD_ALIAS getPropagatorByName ARGS "const char*" name RETURN Propagator
  FUNCTION getPropagator OVERLOAD_ALIAS getPropagatorByIndex ARGS int index RETURN Propagator
  FUNCTION getPropagatorCount RETURN int
  FUNCTION getDistanceQuery OVERLOAD_ALIAS getDistanceQueryByName ARGS "const char*" name RETURN DistanceQuery
)
