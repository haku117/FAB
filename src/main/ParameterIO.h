#pragma once
#include <string>
#include <vector>
#include <iostream>

struct ParameterIO
{
	std::string name;
	std::string param;
	
	ParameterIO(const std::string& name, const std::string& param);

	void write(std::ostream& os, const std::vector<double>& w);
	std::pair<std::string, std::vector<double>> load(std::istream& is);
private:
	// LR
	void writeLR(std::ostream& os, const std::vector<double>& w);
	std::pair<std::string, std::vector<double>> loadLR(std::istream& is);
	// MLP
	void writeMLP(std::ostream& os, const std::vector<double>& w);
	std::pair<std::string, std::vector<double>> loadMLP(std::istream& is);
	// CNN
	void writeCNN(std::ostream& os, const std::vector<double>& w);
	std::pair<std::string, std::vector<double>> loadCNN(std::istream& is);
	// KMeans
	void writeKM(std::ostream& os, const std::vector<double>& w);
	std::pair<std::string, std::vector<double>> loadKM(std::istream& is);
};
