set(HEADERS
	Trainer.h
	GD.h
	EM.h
)
set(SOURCES
	Trainer.cpp
	GD.cpp
	EM.cpp
)
add_library(train
	${HEADERS} ${SOURCES})

target_link_libraries(train
	logging
)