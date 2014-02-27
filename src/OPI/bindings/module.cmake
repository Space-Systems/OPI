
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
  FUNCTION getPropertyType OVERLOAD_ALIAS getPropertyTypeByString ARGS std::string name RETURN PropertyType
  FUNCTION getPropertyType OVERLOAD_ALIAS getPropertyTypeByIndex ARGS int index RETURN PropertyType

  FUNCTION setProperty OVERLOAD_ALIAS setPropertyInt ARGS std::string name int value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyFloat ARGS std::string name float value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyDouble ARGS std::string name double value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyString ARGS std::string name std::string value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyIntArray ARGS std::string name int* value int n
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyFloatArray ARGS std::string name float* value int n
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyDoubleArray ARGS std::string name double* value int n

  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntByName ARGS std::string property RETURN int
  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntByIndex ARGS int index RETURN int
  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntElementByName ARGS std::string property int element RETURN int
  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntElementByIndex ARGS int index int element RETURN int
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatByName ARGS std::string property RETURN float
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatByIndex ARGS int index RETURN float
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatElementByName ARGS std::string property int element RETURN float
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatElementByIndex ARGS int index int element RETURN float
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleByName ARGS std::string property RETURN double
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleByIndex ARGS int index RETURN double
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleElementByName ARGS std::string property int element RETURN double
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleElementByIndex ARGS int index int element RETURN double
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringByName ARGS std::string property RETURN std::string
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringByIndex ARGS int index RETURN std::string
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringElementByName ARGS std::string property int element RETURN std::string
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringElementByIndex ARGS int index int element RETURN std::string

  FUNCTION createProperty OVERLOAD_ALIAS createPropertyInt ARGS std::string name int value
  FUNCTION createProperty OVERLOAD_ALIAS createPropertyFloat ARGS std::string name float value
  FUNCTION createProperty OVERLOAD_ALIAS createPropertyString ARGS std::string name std::string value

  FUNCTION setPrivateData ARGS void* data
  FUNCTION getPrivateData RETURN void*
)

