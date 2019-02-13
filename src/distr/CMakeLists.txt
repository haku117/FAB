set(HEADERS
	IDMapper.h
	IntervalEstimator.h
	Runner.h
	Master.h
	Worker.h
)
set(SOURCES
	IDMapper.cpp
	IntervalEstimator.cpp
	Runner.cpp
	Master.cpp
	Worker.cpp
)
add_library(distr
	${HEADERS} ${SOURCES})
target_link_libraries(distr
	common data math model train util
	driver logging network 
)