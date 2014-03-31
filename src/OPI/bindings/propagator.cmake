DECLARE_CLASS( IndexList )

BIND_CLASS( Propagator
  FUNCTION propagate OVERLOAD_ALIAS propagateAll ARGS ObjectData& data double julian_day float dt RETURN ErrorCode
  FUNCTION propagate OVERLOAD_ALIAS propagateIndexed ARGS ObjectData& data IndexList& list double julian_day float dt RETURN ErrorCode
)
