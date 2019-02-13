set(HEADERS
	norm.h
	activation_func.h
)
set(SOURCES
	norm.cpp
	activation_func.cpp
)
add_library(math
	${HEADERS} ${SOURCES})
