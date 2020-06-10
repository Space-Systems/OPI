
BIND_CLASS( Module
  FUNCTION enable RETURN ErrorCode
  FUNCTION disable RETURN ErrorCode
  FUNCTION setName ARGS "const char*" name
  FUNCTION getName RETURN "const char*"
  FUNCTION setAuthor ARGS "const char*" author
  FUNCTION getAuthor RETURN "const char*"
  FUNCTION setDescription ARGS "const char*" desc
  FUNCTION getDescription RETURN "const char*"

  FUNCTION getPropertyCount RETURN int
  FUNCTION getPropertyName ARGS int index RETURN "const char*"
  FUNCTION getPropertyType OVERLOAD_ALIAS getPropertyTypeByString ARGS "const char*" name RETURN PropertyType
  FUNCTION getPropertyType OVERLOAD_ALIAS getPropertyTypeByIndex ARGS int index RETURN PropertyType

  FUNCTION setProperty OVERLOAD_ALIAS setPropertyInt ARGS "const char*" name int value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyFloat ARGS "const char*" name float value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyDouble ARGS "const char*" name double value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyString ARGS "const char*" name "const char*" value
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyIntArray ARGS "const char*" name int* value int n
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyFloatArray ARGS "const char*" name float* value int n
  FUNCTION setProperty OVERLOAD_ALIAS setPropertyDoubleArray ARGS "const char*" name double* value int n

  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntByName ARGS "const char*" property RETURN int
  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntByIndex ARGS int index RETURN int
  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntElementByName ARGS "const char*" property int element RETURN int
  FUNCTION getPropertyInt OVERLOAD_ALIAS getPropertyIntElementByIndex ARGS int index int element RETURN int
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatByName ARGS "const char*" property RETURN float
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatByIndex ARGS int index RETURN float
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatElementByName ARGS "const char*" property int element RETURN float
  FUNCTION getPropertyFloat OVERLOAD_ALIAS getPropertyFloatElementByIndex ARGS int index int element RETURN float
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleByName ARGS "const char*" property RETURN double
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleByIndex ARGS int index RETURN double
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleElementByName ARGS "const char*" property int element RETURN double
  FUNCTION getPropertyDouble OVERLOAD_ALIAS getPropertyDoubleElementByIndex ARGS int index int element RETURN double
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringByName ARGS "const char*" property RETURN "const char*"
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringByIndex ARGS int index RETURN "const char*"
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringElementByName ARGS "const char*" property int element RETURN "const char*"
  FUNCTION getPropertyString OVERLOAD_ALIAS getPropertyStringElementByIndex ARGS int index int element RETURN "const char*"

  FUNCTION createProperty OVERLOAD_ALIAS createPropertyInt ARGS "const char*" name int value
  FUNCTION createProperty OVERLOAD_ALIAS createPropertyFloat ARGS "const char*" name float value
  FUNCTION createProperty OVERLOAD_ALIAS createPropertyDouble ARGS "const char*" name double value
  FUNCTION createProperty OVERLOAD_ALIAS createPropertyString ARGS "const char*" name "const char*" value

  FUNCTION setPrivateData ARGS void* data
  FUNCTION getPrivateData RETURN void*
)

