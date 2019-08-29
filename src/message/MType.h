#pragma once

typedef int msg_t;

struct MType {
	// Basic Control
	static constexpr int CReply = 0;
	static constexpr int COnline = 1;
	static constexpr int CRegister = 2;
	static constexpr int CWorkers = 3;
	static constexpr int CXLength = 4;
	static constexpr int CTerminate = 7;
	static constexpr int CClosed = 8;
	static constexpr int CAlive = 9;

	// Data and data Request
	static constexpr int DParameter = 10;
	static constexpr int DRParameter = 11;
	static constexpr int DDelta = 15;
	static constexpr int DRDelta = 16;
	static constexpr int DDeltaRPL = 17;
	static constexpr int DReport = 18;
	static constexpr int DDeltaReq = 19;

	// Working Control
	static constexpr int CTrainPause = 20;
	static constexpr int CTrainContinue = 21;
	static constexpr int CTrainInterval = 22;

	// Process and Progress (Termination)
	static constexpr int PApply = 40;
	static constexpr int PSend = 41;
	static constexpr int PReport = 42;
	static constexpr int PRequest = 43;
	static constexpr int PFinish = 44;

	// Staticstics
	static constexpr int SGather = 60;
};