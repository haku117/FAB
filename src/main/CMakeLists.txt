set(HEADERS
	func.h
	ParameterIO.h
)
set(SOURCES
	func.cpp
	ParameterIO.cpp
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
# exe: main
add_executable(main main.cpp
	${HEADERS} ${SOURCES})
target_link_libraries(main
#	common data model train util
	distr train #driver logging network 
	#${Boost_PROGRAM_OPTIONS_LIBRARY}
)
# exe: postprocess
add_executable(postprocess postprocess.cpp
	${HEADERS} ${SOURCES})
target_link_libraries(postprocess
	common data model train util
)
# exe: generate
#add_executable(generate generate.cpp
#	${HEADERS} ${SOURCES})
#target_link_libraries(generate
#	model util
#)