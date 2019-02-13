
set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories( ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(util)
add_subdirectory(math)
add_subdirectory(data)
add_subdirectory(model)
add_subdirectory(train)

add_subdirectory(serial)
add_subdirectory(type_traits)
add_subdirectory(logging)
add_subdirectory(network)
#add_subdirectory(network2)
add_subdirectory(driver)
add_subdirectory(distr)

add_subdirectory(common)
add_subdirectory(main)
