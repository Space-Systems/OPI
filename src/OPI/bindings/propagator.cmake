DECLARE_CLASS( IndexList )

BIND_CLASS( Propagator
  FUNCTION propagate OVERLOAD_ALIAS propagateAll ARGS Population& data double julian_day double dt RETURN ErrorCode
  FUNCTION propagate OVERLOAD_ALIAS propagateIndexed ARGS Population& data IndexList& list double julian_day double dt RETURN ErrorCode
)
