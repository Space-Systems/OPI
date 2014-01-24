
BIND_CLASS( Module
  FUNCTION enable RETURN ErrorCode
  FUNCTION disable RETURN ErrorCode
  FUNCTION setName ARGS std::string name
  FUNCTION getName RETURN std::string
  FUNCTION setAuthor ARGS std::string author
  FUNCTION getAuthor RETURN std::string
  FUNCTION setDescription ARGS std::string desc
  FUNCTION getDescription RETURN std::string
)
