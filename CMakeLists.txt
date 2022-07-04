cmake_minimum_required(VERSION 3.0.2)
project(easyocr_ros)

find_package(
    catkin REQUIRED
    catkin_virtualenv
)

catkin_package()

if (${catkin_virtualenv_VERSION} VERSION_LESS "0.6.1")
  message(WARNING "Please install catkin_virtualenv>=0.6.1.")
  message(WARNING "Current catkin_virtualen version is ${catkin_virtualenv_VERSION}")
else()
  if("$ENV{ROS_DISTRO}" STREQUAL "kinetic")
    file(COPY requirements.txt.kinetic DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
    file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/requirements.txt.kinetic requirements.txt)
    catkin_generate_virtualenv(
      INPUT_REQUIREMENTS requirements.txt
      PYTHON_INTERPRETER python3
      USE_SYSTEM_PACKAGES TRUE
      ISOLATE_REQUIREMENTS FALSE
      CHECK_VENV FALSE
    )
  else()
    catkin_generate_virtualenv(
      INPUT_REQUIREMENTS requirements.in
      PYTHON_INTERPRETER python3
      USE_SYSTEM_PACKAGES TRUE
      ISOLATE_REQUIREMENTS FALSE
      CHECK_VENV FALSE
    )
  endif()
  file(GLOB NODE_SCRIPTS_FILES node_scripts/*)
  catkin_install_python(
    PROGRAMS ${NODE_SCRIPTS_FILES}
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
  )
endif()

install(
  DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  USE_SOURCE_PERMISSIONS
)

install(FILES requirements.in requirements.txt
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
