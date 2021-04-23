DECLARE_CLASS( IndexList )

BIND_CLASS( Propagator
  FUNCTION propagate OVERLOAD_ALIAS propagate ARGS Population& population JulianDay epoch long dt PropagationMode mode IndexList* indices RETURN ErrorCode
)
