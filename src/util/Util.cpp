#include "Util.h"
#include <vector>
#include "math.h"
#include <algorithm>

#if defined(_WIN32) || defined(_WIN64)
#define WIN32_LEAN_AND_MEAN             // Exclude rarely-used stuff from Windows headers
#define _OS_WIN
#include <windows.h>
#else
#include <sys/ioctl.h>
#include <unistd.h>
#endif

using namespace std;

std::pair<int, int> getScreenSize() {
	int cols = -1;
	int lines = -1;

#ifdef _OS_WIN
//	printf("win\n");
	CONSOLE_SCREEN_BUFFER_INFO csbi;
	GetConsoleScreenBufferInfo(GetStdHandle(STD_OUTPUT_HANDLE), &csbi);
	cols = csbi.dwSize.X;
	lines = csbi.dwSize.Y;
#else
//	printf("posix\n");
	struct winsize ts;
	ioctl(STDIN_FILENO, TIOCGWINSZ, &ts);
	cols = ts.ws_col;
	lines = ts.ws_row;
#endif 
	return make_pair(cols, lines);
}

#undef _OS_WIN

bool beTrueOption(const std::string& str){
	static vector<string> true_options({"1", "t", "T", "true", "True", "TRUE", "y", "Y", "yes", "Yes", "YES"});
	return find(true_options.begin(), true_options.end(), str) != true_options.end();
}


std::vector<int> getIntList(const std::string & str, const std::string& sepper)
{
	std::vector<int> res;
	size_t pl = 0;
	size_t p = str.find_first_of(sepper);
	while(p != string::npos){
		res.push_back(stoi(str.substr(pl, p - pl)));
		pl = p + 1;
		p = str.find_first_of(sepper, pl);
	}
	if(!str.empty() && pl < str.size())
		res.push_back(stoi(str.substr(pl)));
	return res;
}

std::vector<double> getDoubleList(const std::string & str, const std::string& sepper)
{
	std::vector<double> res;
	size_t pl = 0;
	size_t p = str.find_first_of(sepper);
	while(p != string::npos){
		res.push_back(stod(str.substr(pl, p - pl)));
		pl = p + 1;
		p = str.find_first_of(sepper, pl);
	}
	if(!str.empty() && pl < str.size())
		res.push_back(stod(str.substr(pl)));
	return res;
}

std::vector<std::string> getStringList(const std::string & str, const std::string& sepper)
{
	std::vector<std::string> res;
	size_t pl = 0;
	size_t p = str.find_first_of(sepper);
	while(p != string::npos){
		res.push_back(str.substr(pl, p - pl));
		pl = p + 1;
		p = str.find_first_of(sepper, pl);
	}
	if(!str.empty() && pl < str.size())
		res.push_back(str.substr(pl));
	return res;
}

int stoiKMG(const std::string & str, const bool binary)
{
	if(str.empty())
		return 0;
	const int f = binary ? 1024 : 1000;
	char ch = str.back();
	int factor = 1;
	if(ch == 'k' || ch == 'K')
		factor = f;
	else if(ch == 'm' || ch == 'M')
		factor = f * f;
	else if(ch == 'g' || ch == 'G')
		factor = f * f * f;
	return stoi(str)*factor;
}

size_t stoulKMG(const std::string & str, const bool binary)
{
	if(str.empty())
		return 0;
	const size_t f = binary ? 1024 : 1000;
	char ch = str.back();
	size_t factor = 1;
	if(ch == 'k' || ch == 'K')
		factor = f;
	else if(ch == 'm' || ch == 'M')
		factor = f * f;
	else if(ch == 'g' || ch == 'G')
		factor = f * f * f;
	return stoul(str)*factor;
}


void accumuteDeltaSave(std::vector<double>& delta, std::vector<double>& d){

	int rank = d.size()/2 - 1;
	int indxWu = int(d[0]);
	int indxHi = int(d[rank]);
	for(size_t j = 0; j < delta.size(); j += rank+1){
		if(delta[j] == indxWu) {// match the dimension of delta
			for(int k = 0; k < rank; k++){
				delta[j+1 + k] += d[1 + k]; 
			}
			indxWu = -1;
		}
		else if(delta[j] == indxHi){
			for(int k = 0; k < rank; k++){
				delta[j+1 + k] += d[rank+2 + k]; 
			}
			indxHi = -1;
		}
	}
	if(indxWu != -1) {
		delta.insert(delta.end(), d.begin(), d.begin()+rank+1);
	}
	if(indxHi != -1) {
		delta.insert(delta.end(), d.begin()+rank+1, d.end());
	}
}

std::vector<int> parseParam(const std::string& param){

	size_t pstart = 0, pend = 0;
	std::vector<int> tokens;
	std::string delimiter = ",";
	while ((pend = param.find(delimiter, pstart)) != std::string::npos) {
    	tokens.push_back(stoi(param.substr(pstart, pend)));
    	pstart = pend + delimiter.length();
	}
    tokens.push_back(stoi(param.substr(pstart)));
	// if(tokens.size() != 3){
	// 	cout << "incorrect params for NMF: " << tokens.size();
	// }
	return tokens;
}