set(HEADERS
	Timer.h
	#FileEnumerator.h
	Util.h
	Matrix.h
)
set(SOURCES
	Timer.cpp
	#FileEnumerator.cpp
	Util.cpp
	Matrix.cpp
)
add_library(util
	${HEADERS} ${SOURCES})