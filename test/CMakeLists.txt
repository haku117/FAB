include_directories("../src/")

add_custom_target(mytest DEPENDS
	data-load train-simple mw-simple mw-thread communication unit-worker
	model-lr model-mlp model-cnn)

add_executable(data-load data-load.cpp)
target_link_libraries(data-load data)

add_executable(train-simple train-simple.cpp)
target_link_libraries(train-simple data model train)

add_executable(mw-simple mw-simple.cpp)
target_link_libraries(mw-simple data model train)

add_executable(mw-thread mw-thread.cpp)
target_link_libraries(mw-thread data model train ${CMAKE_THREAD_LIBS_INIT})

add_executable(communication communication.cpp)
target_include_directories(communication PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(communication network logging util
	${MPI_CXX_LIBRARIES}
)

add_executable(unit-worker unit-worker.cpp)
target_include_directories(unit-worker PRIVATE ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(unit-worker distr #data model train common driver network logging 
	${CMAKE_THREAD_LIBS_INIT}
	${MPI_CXX_LIBRARIES}
)

add_executable(model-lr model-lr.cpp)
target_link_libraries(model-lr data model train util logging)

add_executable(model-mlp model-mlp.cpp)
target_link_libraries(model-mlp data model train util logging)

add_executable(model-cnn model-cnn.cpp)
target_link_libraries(model-cnn data model train util logging)
