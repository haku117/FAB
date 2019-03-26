#include "KernelFactory.h"
#include "LogisticRegression.h"
#include "MLP.h"
#include "CNN.h"
#include "Kmeans.h"
#include "NMF.h"
#include <stdexcept>
using namespace std;

Kernel* KernelFactory::generate(const std::string& name){
	if(name == "lr")
	 	return new LogisticRegression();
	else if(name == "mlp")
	 	return new MLP();
	else if(name == "cnn")
		return new CNN();
	else if(name == "km")
		return new Kmeans();
	else if(name == "nmf")
		return new NMF();
	else
		throw invalid_argument("do not support the method: " + name);
	return nullptr;
}
