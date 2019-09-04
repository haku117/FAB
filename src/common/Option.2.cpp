#include "Option.h"
#include <vector>
#include <iostream>
#include <algorithm>
#include <boost/program_options.hpp>
#include "util/Util.h"
#include "data/DataLoader.h"
#include "model/KernelFactory.h"
#include "train/TrainerFactory.h"

using namespace std;

struct Option::Impl{
	boost::program_options::options_description desc;
	Impl(const int width):desc("Options", width)
	{}
};

static string VecToString(const vector<string>& vec){
	string res;
	for(auto& v : vec)
		res += v + " ";
	if(!vec.empty())
		res.pop_back();
	return res;
}

bool Option::parse(int argc, char * argv[], const size_t nWorker)
{
	string tmp_cast;
	string tmp_interval;
	string tmp_ids, tmp_idy;
	string tmp_bs;
	string tmp_t_iter, tmp_a_iter, tmp_l_iter;
	int tmp_v;
	if(pimpl == nullptr)
		pimpl = new Impl(getScreenSize().first);

	string desc_alg = "The algorithm/model to run/train. "
		"Support: " + VecToString(KernelFactory::supportList());
	string desc_opt = "The optimizer to adopt. "
		"Support: " + VecToString(TrainerFactory::supportList());
	string desc_dl = "The dataset/type to use. Give dataset name (mnist) or "
		"type (csv, tsv, customize, list (variable length x)). "
		"Support: " + VecToString(DataLoader::supportList());;

	using boost::program_options::value;
	using boost::program_options::bool_switch;
	pimpl->desc.add_options()
		("help,h", "Print help messages")
		// parallel
		("mode,m", value(&conf.mode)->default_value("bsp"), "The parallel mode: bsp, tap, ssp:<n>, sap:<n>, fsp, aap")
		// parallel - broadcast
		("cast_mode,c", value(&tmp_cast)->default_value("broadcast"),
			"The method to send out new parameters. Supports: broadcast/all, ring:k, random:k,seed, hash:k")
		// parallel - fsp
		("flex_interval", value(&tmp_interval)->default_value("portion:0.05"),
			"The method to decide update interval for FSP. "
			"Supports: interval:x(x is in seconds), portion:x(x in 0~1), "
			"improve:x,t (x: avg. imporovement, t: max waiting time), balance:w (num. of windows)")
		// app - algorithm
		("algorithm,a", value(&conf.algorighm)->required(), desc_alg.c_str())
		("parameter,p", value(&conf.algParam)->required(),
			"The parameter of the algorithm, usually the shape of the algorithm")
		("seed", value(&conf.seed)->default_value(123456U), "The seed to initialize parameters")
		// app - training
		("batch_size,s", value(&tmp_bs)->required(), "The global batch size. Support suffix: k, m, g")
		//("learning_rate,l", value(&conf.lrate)->required(), "The learning rate")
		("optimizer,o", value(&conf.optimizer)->required()->default_value("gd:0.01"), desc_opt.c_str())
		// file - input
		("dataset", value(&conf.dataset)->default_value("csv"), desc_dl.c_str())
		("trainpart", bool_switch(&conf.trainPart)->default_value(true),
			"Whether to use the trainning part of the dateset.")
		("data_file,d", value(&conf.fnData)->required(), "The file name of the input data.")
		("topk,k", value(&conf.topk)->default_value(0), "Only use the first <k> data points. (0 means all)")
		("normalize,n", bool_switch(&conf.normalize)->default_value(false),
			"Whether to do normailzation on the input file.")
		("shuffle", bool_switch(&conf.shuffle)->default_value(false), "Randomly shuffle the dataset.")
		// file - input - table
		("header", bool_switch(&conf.header)->default_value(false), 
			"Whether the input file contain a header line")
		("sepper", value(&conf.sepper)->default_value(","), "Separator for customized dataset.")
		("skip", value(&tmp_ids)->default_value({}, ""),
			"The columns to skip in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)")
		// file - input - list (variable-length x)
		("unit", value(&conf.lenUnit)->default_value(0), "Length of one x unit for the variable length input (RNN).")
		("ylist,y", value(&tmp_idy)->default_value({}, ""),
			"The columns to be used as y in the data file. "
			"A space/comma separated list of integers and a-b (a, a+1, a+2, ..., b)")
		// file - output
		("record_file,r", value(&conf.fnOutput)->required(), "The file name of the archived parameter.")
		("binary,b", bool_switch(&conf.binary)->default_value(false), "Whether to output using binary IO.")
		("resume", bool_switch(&conf.resume)->default_value(false), "Whether to resume from the last output item.")
		// termination
		("term_iter", value(&tmp_t_iter)->required(), "Termination condition: maximum iteration.")
		("term_time", value(&conf.tcTime)->required(), "Termination condition: maximum training time.")
		// archive
		("arch_iter", value(&tmp_a_iter)->default_value("1000"), "Progress archiving condition: maximum iteration.")
		("arch_time", value(&conf.arvTime)->default_value(1.0), "Progress archiving condition: maximum training time.")
		// log
		("log_iter", value(&tmp_l_iter)->default_value("100"), "Log training step every <log_iter> iterations.")
		("v", value(&tmp_v), "Verbose level.")
		;

	boost::program_options::variables_map vm;
	conf.nw = nWorker;
	try {
		auto p = boost::program_options::command_line_parser(argc, argv).
			//options(pimpl->desc).allow_unregistered().run();
			options(pimpl->desc).run();
		boost::program_options::store(p, vm);
		boost::program_options::notify(vm);

		conf.mcastParam = getStringList(tmp_cast, ":,; ");
		conf.intervalParam = getStringList(tmp_interval, ":,; ");
		conf.idSkip = getIntListByRange(tmp_ids);
		conf.idY = getIntListByRange(tmp_idy);
		conf.batchSize = stoiKMG(tmp_bs);
		conf.tcIter = stoiKMG(tmp_t_iter);
		conf.arvIter = stoiKMG(tmp_a_iter);
		conf.logIter = stoiKMG(tmp_l_iter);
	} catch(exception& e){
		cerr << "Error in parsing parameter: " << e.what() << endl;
		return false;
	}
	if(vm.count("help") != 0)
		return false;

	// check and preprocess
	if(!processMode()){
		cerr << "Error: mode not supported: " << conf.mode << endl;
		return false;
	}
	if(!processDataset()){
		cerr << "Error: dataset not supported: " << conf.dataset << endl;
		return false;
	}
	if(!processAlgorithm()){
		cerr << "Error: algorithm not supported: " << conf.algorighm << endl;
		return false;
	}
	if(!processOptimizer()){
		cerr << "Error: optimizer not supported: " << conf.optimizer << endl;
		return false;
	}
	// error handling

	return true;
}

void Option::showUsage() const {
	if(pimpl == nullptr)
		return;
	cout << pimpl->desc << endl;
}

bool Option::processMode(){
	for(char& ch : conf.mode){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> t = getStringList(conf.mode, ":-, ");
	vector<string> supported = { "bsp", "tap", "ssp", "sap", "fsp", "aap" };
	auto it = find(supported.begin(), supported.end(), t[0]);
	if(it == supported.end())
		return false;
	conf.mode = t[0];
	if(t[0] == "ssp" || t[0] == "sap"){
		if(t.size() > 1)
			conf.staleGap = stoi(t[1]);
		else
			conf.staleGap = 1;
	} else if(t[0]=="aap"){
		conf.aapWait = t.size() >= 2 && beTrueOption(t[1]);
	}
	return true;
}

bool Option::processDataset(){
	for(char& ch : conf.dataset){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	return DataLoader::isSupported(conf.dataset);
}

bool Option::processAlgorithm(){
	for(char& ch : conf.algorighm){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	return KernelFactory::isSupported(conf.algorighm);
}

bool Option::processOptimizer()
{
	for(char& ch : conf.optimizer){
		if(ch >= 'A' && ch <= 'Z')
			ch += 'a' - 'A';
	}
	vector<string> t = getStringList(conf.optimizer, ":;, ");
	if(t.empty() || !TrainerFactory::isSupported(t[0]))
		return false;
	conf.optimizer = t[0];
	for(size_t i = 1; i < t.size(); ++i)
		conf.optimizerParam.push_back(t[i]);
	return true;
}

