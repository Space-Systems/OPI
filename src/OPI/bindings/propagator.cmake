DECLARE_CLASS( IndexList )

BIND_CLASS( Propagator
  FUNCTION propagate OVERLOAD_ALIAS propagateAll ARGS Population& population double julian_day double dt PropagationMode mode IndexList* indices RETURN ErrorCode
)
