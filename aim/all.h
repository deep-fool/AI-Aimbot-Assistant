#pragma once
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN  // 阻止 windows.h 自动包含 winsock.h
#endif
#include "kmboxNet.h"
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>        // 必须在 winsock2.h 之后
#include "ScreenCapture.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>
#include <memory>
#include <atomic>
#include <thread>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <NvInfer.h>
#include <cmath>
#include "findcolor.h"
#include "Detector.h"
#include "pid.h"
#ifndef NOMINMAX
#define NOMINMAX
#endif
#pragma comment(lib, "winmm.lib")