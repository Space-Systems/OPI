! Test functions for the OPI interface
! Can be compiled into a shared object and called from C or Fortran

subroutine OPI_Plugin_info(info) bind(c, name="OPI_Plugin_info")
  use OPI
  use ISO_C_BINDING
  type(c_ptr), value :: info
  ! init plugin variables
  call OPI_PluginInfo_init(info, OPI_API_VERSION_MAJOR, OPI_API_VERSION_MINOR, &
                           0, 1, 0, OPI_PROPAGATOR_PLUGIN)
  call OPI_PluginInfo_setName(info, "Fortran Example Propagator")
  call OPI_PluginInfo_setAuthor(info, "ILR TU BS")
  call OPI_PluginInfo_setDescription(info, "A non-functional Fortran Propagator")
end subroutine

function OPI_Plugin_propagate( propagator, data, years, seconds, dt) result(error_code) bind(c, name="OPI_Plugin_propagate")
  use OPI
  use OPI_Types
  use ISO_C_BINDING
  integer(c_int) :: error_code
  type(c_ptr), value :: data
  type(c_ptr), value :: propagator
  real(c_float) :: years
  real(c_float) :: seconds
  real(c_float) :: dt
  integer(c_int) :: size
  type(OPI_Orbit), dimension(:), pointer :: orbit

  write(*,*) 'Running Fortran Propagator'

  orbit => OPI_ObjectData_getOrbit(data)

  size = OPI_ObjectData_getSize(data)

  do 20 i = 1, size, 1
    orbit(i)%inclination = i
  20  continue

  error_code = OPI_ObjectData_update(data, OPI_DATA_ORBIT);

  write(*,*) 'done'

  error_code = OPI_NO_ERROR
end function
