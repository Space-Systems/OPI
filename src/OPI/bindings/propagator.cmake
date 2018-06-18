DECLARE_CLASS( IndexList )

BIND_CLASS( Propagator
  FUNCTION propagate OVERLOAD_ALIAS propagateAll ARGS Population& population double julian_day double dt RETURN ErrorCode
  FUNCTION propagate OVERLOAD_ALIAS propagateIndexed ARGS Population& population IndexList& list double julian_day double dt RETURN ErrorCode
)
