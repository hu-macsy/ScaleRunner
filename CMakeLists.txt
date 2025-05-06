# Set minimum version of CMake.
cmake_minimum_required(VERSION 3.11.4)

# Set project name and version
project(scalerunner VERSION 1.0.0)

include(GNUInstallDirs)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(scalerunner STATIC)

set(public_headers
  include/scalerunner/first_order_rw.h
  include/scalerunner/graph.h
  include/scalerunner/node2vec.h
  include/scalerunner/random_walk.h
  include/scalerunner/rw_scheduler.h
  include/scalerunner/second_order_rw.h
  include/scalerunner/utils.h
  include/scalerunner/mpi_utils.h
)
  
target_sources(${PROJECT_NAME}
  PRIVATE
    src/mpi_utils.cpp
)

find_package(OpenMP)
target_link_libraries(${PROJECT_NAME} INTERFACE OpenMP::OpenMP_CXX)

find_package(dhb REQUIRED)
target_link_libraries(scalerunner PUBLIC dhb::dhb)

find_package(MPI REQUIRED)
target_link_libraries(scalerunner PRIVATE MPI::MPI_CXX)

set_property(TARGET scalerunner PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME}
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

install(TARGETS ${PROJECT_NAME}
    EXPORT "${PROJECT_NAME}Targets"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${public_headers} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME})

install(EXPORT "${PROJECT_NAME}Targets"
  FILE "${PROJECT_NAME}Targets.cmake"
  NAMESPACE scalerunner::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

add_library(${PROJECT_NAME}::${PROJECT_NAME} STATIC IMPORTED)

set_target_properties(${PROJECT_NAME}::${PROJECT_NAME} PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_IMPORT_PREFIX}/include"
)

set_target_properties(${PROJECT_NAME} PROPERTIES PUBLIC_HEADER "${public_headers}")
set_target_properties(${PROJECT_NAME} PROPERTIES DEBUG_POSTFIX "d")

# ===================================
# Make a Configuration Package
# ===================================
include(CMakePackageConfigHelpers)

export(EXPORT ${PROJECT_NAME}Targets
       FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Targets.cmake"
       NAMESPACE ${PROJECT_NAME}::
)

configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  VERSION "${version}"
  COMPATIBILITY AnyNewerVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}
)

set_property(TARGET scalerunner PROPERTY VERSION ${version})
set_property(TARGET scalerunner PROPERTY SOVERSION 0)
set_property(TARGET scalerunner PROPERTY
  INTERFACE_scalerunner_MAJOR_VERSION 1)
set_property(TARGET scalerunner APPEND PROPERTY
  COMPATIBLE_INTERFACE_STRING scalerunner_MAJOR_VERSION
)

# ========================
# Test Target
# ========================
option(SR_BUILD_TESTS "Build test target." OFF)
option(SR_BUILD_MPI_TESTS "Build mpi test target." OFF)
if (SR_BUILD_TESTS)
  if(EXISTS "${PROJECT_SOURCE_DIR}/test/lib/Catch2")
    add_subdirectory(${PROJECT_SOURCE_DIR}/test/lib/Catch2)
  else()
    message(FATAL_ERROR
      "Missing Catch2 library in test/lib/Catch2 "
      "Please run `git submodule update --init` to fetch the submodule.")
  endif()

  if(EXISTS "${PROJECT_SOURCE_DIR}/test/graphs")
    add_definitions(-DSR_TEST_GRAPH_DIR="${PROJECT_SOURCE_DIR}/test/graphs")
  else()
    message(FATAL_ERROR
      "Could not find test graph directory.")
  endif()

  add_executable(scalerunner_test
    test/1st_order_rw_test.cpp
    test/2nd_order_rw_test.cpp
    test/graph_reading_test.cpp
    test/mockup_test.cpp
    test/node2vec_unweighted_test.cpp
    test/node2vec_weighted_test.cpp
    test/rw_consecutive_paths_test.cpp
    test/rw_scheduler_test.cpp
    test/simple_rw_test.cpp
    test/utils_test.cpp
    test/weighted_rw_test.cpp
  )

  target_link_libraries(scalerunner_test PRIVATE scalerunner)

  find_package(gdsb REQUIRED)
  target_link_libraries(scalerunner_test PRIVATE gdsb::gdsb)

  # find_package(Catch2 3 REQUIRED)
  target_link_libraries(scalerunner_test PRIVATE Catch2::Catch2WithMain)
  
  install(
    TARGETS scalerunner_test
    RUNTIME DESTINATION bin
    )
    
  # Test setup. Use CTest + Catch2's CTest integration.
  include(CTest)
  include(Catch)
  catch_discover_tests(scalerunner_test)
    
    # GDSB MPI test target
  if (SR_BUILD_MPI_TESTS)
    add_executable(scalerunner_mpi_test
      test/mpi_test_main.cpp
      test/mpi_test.cpp
    )
    
    # Debugging Libraries
    # target_link_libraries(gdsb_mpi_test PRIVATE asan)
    
    target_link_libraries(scalerunner_mpi_test PRIVATE scalerunner)
    
    target_link_libraries(scalerunner_mpi_test PRIVATE gdsb MPI::MPI_CXX)
    
    target_link_libraries(scalerunner_mpi_test PRIVATE Catch2::Catch2)

    install(
      TARGETS scalerunner_mpi_test
      RUNTIME DESTINATION bin
    )

    catch_discover_tests(scalerunner_mpi_test)
  endif()

endif()

# ========================
# Demo Target
# ========================
option(SR_DEMO "Build demo program." OFF)
if(SR_DEMO)
  add_executable(scalerunner_demo
    demo/demo.cpp
  )

  target_link_libraries(scalerunner_demo PRIVATE scalerunner)

  find_package(gdsb REQUIRED)
  target_link_libraries(scalerunner_demo PRIVATE gdsb::gdsb)

  install(
    TARGETS scalerunner_demo
    RUNTIME DESTINATION bin
  )
endif()

if (SR_DEMO OR SR_BUILD_TESTS)
  if(EXISTS "${PROJECT_SOURCE_DIR}/test/graphs")
  add_definitions(-DSR_TEST_GRAPH_DIR="${PROJECT_SOURCE_DIR}/test/graphs")
  else()
  message(FATAL_ERROR
    "Could not find test graph directory.")
  endif()
endif()