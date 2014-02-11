
BIND_CLASS( Module
  FUNCTION enable RETURN ErrorCode
  FUNCTION disable RETURN ErrorCode
  FUNCTION setName ARGS std::string name
  FUNCTION getName RETURN std::string
  FUNCTION setAuthor ARGS std::string author
  FUNCTION getAuthor RETURN std::string
  FUNCTION setDescription ARGS std::string desc
  FUNCTION getDescription RETURN std::string

  FUNCTION getPropertyCount RETURN int
  FUNCTION getPropertyName ARGS int index RETURN std::string

  FUNCTION setProperty OVERLOAD_ALIAS setPropertyInt ARGS std::string name int value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyFloat ARGS std::string name float value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyString ARGS std::string name std::string value

  FUNCTION getPropertyInt ARGS std::string property RETURN int
  FUNCTION getPropertyFloat ARGS std::string property RETURN float
  FUNCTION getPropertyString ARGS std::string property RETURN std::string

  FUNCTION createProperty OVERLOAD_ALIAS createPropertyInt ARGS std::string name int value
  FUNCTION createProperty OVERLOAD_ALIAS createPropertyFloat ARGS std::string name float value
  FUNCTION createProperty OVERLOAD_ALIAS createPropertyString ARGS std::string name std::string value

  FUNCTION setPrivateData ARGS void* data
  FUNCTION getPrivateData RETURN void*
)
