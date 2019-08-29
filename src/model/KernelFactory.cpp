#include "KernelFactory.h"
#include "LogisticRegression.h"
#include "MLP.h"
#include "CNN.h"
#include "Kmeans.h"
#include "NMF.h"
#include "LDA.h"
#include "GMM.h"
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
	else if(name.find("nmf") !=std::string::npos)
		return new NMF();
	else if(name.find("lda") !=std::string::npos)
		return new LDA();
	else if(name.find("gmm") !=std::string::npos)
		return new GMM();
	else
		throw invalid_argument("do not support the method: " + name);
	return nullptr;
}
