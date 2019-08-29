#pragma once
#include <chrono>

class Timer
{
	std::chrono::system_clock::time_point _time;
	static std::chrono::system_clock::time_point _boot_time;
public:
	Timer();
	void restart();

	// return millisecond as default
	long long elapse() const {
		return elapseMS();
	}
	long long elapseMS() const;
	int elapseS() const;
	double elapseSd() const;
	double elapseMin() const;

	static double Now();
	static double NowSinceBoot();

	void sleep(double seconds);
};
