#pragma once
#define NOMINMAX
#include "PIDController.h"

// ============================================================
//  MoveMethod 枚举 —— 在这里统一定义，pid.cpp/main.cpp/WebUIBridge.h 共用
// ============================================================
enum class MoveMethod { KMBOX = 0, SYSTEM = 1 };
extern MoveMethod g_moveMethod;   // 定义在 pid.cpp，所有文件可读

// ============================================================
//  函数声明
// ============================================================
void SendData(int x, int y, bool fireNow, float sendOffset, int boxW);
void CenterToScreen(float x, float y);
void SetMoveMethod(int method);   // 0=KMBox, 1=系统SendInput
void ApplyPIDParams();

// ============================================================
//  可调参数 extern 声明（定义在 pid.cpp）
// ============================================================
extern float CFG_KP;
extern float CFG_KI;
extern float CFG_KD;

extern float CFG_PRED_ORDER;
extern float CFG_PRED_WEIGHT_X;
extern float CFG_PRED_WEIGHT_Y;
extern float CFG_PRED_ZERO_DIST;
extern float CFG_PRED_FULL_DIST;

extern float CFG_INIT_SCALE;
extern float CFG_RAMP_MS;
extern float CFG_OUT_MAX;
extern float CFG_DZ_DIST;

extern float CFG_FILTER_ALPHA_NEAR;
extern float CFG_FILTER_NEAR_DIST;

extern float CFG_OSC_DECAY;
extern float CFG_OSC_NEAR_DIST;
extern int   CFG_OSC_STABLE_FRAMES;
extern bool  CFG_OSC_ENABLED;

extern float CFG_NEAR_DIST;
extern float CFG_MID_DIST;
extern float CFG_FAR_DIST;
extern int   CFG_MAX_NEAR;
extern int   CFG_MAX_MID;
extern int   CFG_MAX_FFAR;
extern int   CFG_MAX_FAR;

extern float CFG_OFFSET_RAMP;
extern float CFG_INTEGRAL_CLEAR_DIST;

extern float CFG_BEZ_DIST_THRESH;
extern int   CFG_BEZ_MIN_DELAY;
extern int   CFG_BEZ_MAX_DELAY;
extern int   CFG_BEZ_DURATION_MS;
extern int   CFG_BEZ_STEP_MS;
extern float CFG_BEZ_DEVIATION;
extern int   CFG_BEZ_DIRECT_THRESH;
extern bool  CFG_BEZ_ENABLED;
