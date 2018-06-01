cmake_minimum_required(VERSION 2.8.3)
project(rovioli_deep_reloc)

find_package(catkin_simple REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)

#############
# LIBRARIES #
#############
cs_add_library(${PROJECT_NAME}_lib
  src/data-publisher-flow.cc
  src/localizer.cc
  src/localizer-flow.cc
  src/localizer-helpers.cc
  src/rovioli-node.cc
)

#######
# APP #
#######
cs_add_executable(rovioli
  app/rovioli-app.cc
)
target_link_libraries(rovioli ${PROJECT_NAME}_lib)

#########
# SHARE #
#########
catkin_add_env_hooks(10.rovio-config-template
  SHELLS sh
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/env-hooks
)

############
## EXPORT ##
############
cs_install()
cs_export()
