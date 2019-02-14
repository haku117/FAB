set(HEADERS
	Kernel.h
	KernelFactory.h
	Parameter.h
	Model.h
	#LogisticRegression.h
	#MLP.h
	#MLPProxy.h
	#CNN.h
	#CNNProxy.h
	Kmeans.h
)
set(SOURCES
	Kernel.cpp
	KernelFactory.cpp
	Parameter.cpp
	Model.cpp
	#LogisticRegression.cpp
	#MLP.cpp
	#MLPProxy.cpp
	#CNN.cpp
	#CNNProxy.cpp
	Kmeans.cpp
)
add_library(model
	${HEADERS} ${SOURCES})
target_link_libraries(model math data util)
