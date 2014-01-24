DECLARE_CLASS( IndexList )

BIND_CLASS( Propagator
  FUNCTION propagate OVERLOAD_ALIAS propagateAll ARGS ObjectData& data float years float seconds float dt RETURN ErrorCode
  FUNCTION propagate OVERLOAD_ALIAS propagateIndexed ARGS ObjectData& data IndexList& list float years float seconds float dt RETURN ErrorCode
)
