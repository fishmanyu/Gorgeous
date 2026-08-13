#----------------------------------------------------------------
# Generated CMake target import file for configuration "RelWithDebInfo".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "TBB::tbb" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbb APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbb PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libtbb.so.12.19"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbb.so.12"
  )

list(APPEND _cmake_import_check_targets TBB::tbb )
list(APPEND _cmake_import_check_files_for_TBB::tbb "${_IMPORT_PREFIX}/lib/libtbb.so.12.19" )

# Import target "TBB::tbbmalloc" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbbmalloc APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbbmalloc PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libtbbmalloc.so.2.19"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbbmalloc.so.2"
  )

list(APPEND _cmake_import_check_targets TBB::tbbmalloc )
list(APPEND _cmake_import_check_files_for_TBB::tbbmalloc "${_IMPORT_PREFIX}/lib/libtbbmalloc.so.2.19" )

# Import target "TBB::tbbmalloc_proxy" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbbmalloc_proxy APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbbmalloc_proxy PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELWITHDEBINFO "TBB::tbbmalloc"
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libtbbmalloc_proxy.so.2.19"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbbmalloc_proxy.so.2"
  )

list(APPEND _cmake_import_check_targets TBB::tbbmalloc_proxy )
list(APPEND _cmake_import_check_files_for_TBB::tbbmalloc_proxy "${_IMPORT_PREFIX}/lib/libtbbmalloc_proxy.so.2.19" )

# Import target "TBB::tbbbind_2_5" for configuration "RelWithDebInfo"
set_property(TARGET TBB::tbbbind_2_5 APPEND PROPERTY IMPORTED_CONFIGURATIONS RELWITHDEBINFO)
set_target_properties(TBB::tbbbind_2_5 PROPERTIES
  IMPORTED_LOCATION_RELWITHDEBINFO "${_IMPORT_PREFIX}/lib/libtbbbind_2_5.so.3.19"
  IMPORTED_SONAME_RELWITHDEBINFO "libtbbbind_2_5.so.3"
  )

list(APPEND _cmake_import_check_targets TBB::tbbbind_2_5 )
list(APPEND _cmake_import_check_files_for_TBB::tbbbind_2_5 "${_IMPORT_PREFIX}/lib/libtbbbind_2_5.so.3.19" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
