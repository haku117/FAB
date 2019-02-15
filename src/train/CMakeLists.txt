set(HEADERS
	Trainer.h
	GD.h
)
set(SOURCES
	Trainer.cpp
	GD.cpp
)
add_library(train
	${HEADERS} ${SOURCES})

target_link_libraries(train
	logging
)