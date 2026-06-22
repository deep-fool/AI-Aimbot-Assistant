#include "pid.h"
#include "PIDController.h"
#include <math.h>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#define WIN32_LEAN_AND_MEAN
#include <winsock2.h>
#include <windows.h>
#include "kmboxNet.h"

// ============================================================
//  MoveMethod —— 定义在此，pid.h 里 extern 声明供外部使用
// ============================================================
MoveMethod g_moveMethod = MoveMethod::KMBOX;

void SetMoveMethod(int method) {
    g_moveMethod = (method == 1) ? MoveMethod::SYSTEM : MoveMethod::KMBOX;
}

static inline void DoMouseMove(int dx, int dy) {
    if (dx == 0 && dy == 0) return;
    if (g_moveMethod == MoveMethod::SYSTEM) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dx = dx;
        input.mi.dy = dy;
        input.mi.dwFlags = MOUSEEVENTF_MOVE;
        SendInput(1, &input, sizeof(INPUT));
    }
    else {
        kmNet_enc_mouse_move((short)dx, (short)dy);
    }
}

static inline void DoMouseLeft(int state) {
    if (g_moveMethod == MoveMethod::SYSTEM) {
        INPUT input = {};
        input.type = INPUT_MOUSE;
        input.mi.dwFlags = state ? MOUSEEVENTF_LEFTDOWN : MOUSEEVENTF_LEFTUP;
        SendInput(1, &input, sizeof(INPUT));
    }
    else {
        kmNet_enc_mouse_left(state);
    }
}

// ============================================================
//  屏幕参数（默认从系统自动读取，CenterToScreen 可覆盖）
// ============================================================
static int CENTER_X = 0;  // 0=首次使用时自动读取系统分辨率
static int CENTER_Y = 0;

static int GetCenterX() {
    if (CENTER_X <= 0) CENTER_X = GetSystemMetrics(SM_CXSCREEN) / 2;
    return CENTER_X;
}
static int GetCenterY() {
    if (CENTER_Y <= 0) CENTER_Y = GetSystemMetrics(SM_CYSCREEN) / 2;
    return CENTER_Y;
}

void CenterToScreen(float x, float y) {
    CENTER_X = (int)x;
    CENTER_Y = (int)y;
}

// ============================================================
//  全局可调参数（去掉 const，UI 可直接写入）
// ============================================================
float CFG_KP = 0.20f;
float CFG_KI = 0.018f;
float CFG_KD = 0.002f;

float CFG_PRED_ORDER = 2.0f;
float CFG_PRED_WEIGHT_X = 1.2f;
float CFG_PRED_WEIGHT_Y = 0.3f;
float CFG_PRED_ZERO_DIST = 25.0f;
float CFG_PRED_FULL_DIST = 80.0f;

float CFG_INIT_SCALE = 0.75f;
float CFG_RAMP_MS = 150.0f;
float CFG_OUT_MAX = 150.0f;
float CFG_DZ_DIST = 3.0f;

float CFG_FILTER_ALPHA_NEAR = 0.18f;
float CFG_FILTER_NEAR_DIST = 80.0f;

float CFG_OSC_DECAY = 0.20f;
float CFG_OSC_NEAR_DIST = 70.0f;
int   CFG_OSC_STABLE_FRAMES = 4;
bool  CFG_OSC_ENABLED = true;

float CFG_NEAR_DIST = 20.0f;
float CFG_MID_DIST = 45.0f;
float CFG_FAR_DIST = 90.0f;
int   CFG_MAX_NEAR = 3;
int   CFG_MAX_MID = 12;
int   CFG_MAX_FFAR = 32;
int   CFG_MAX_FAR = 110;

float CFG_OFFSET_RAMP = 30.0f;
float CFG_INTEGRAL_CLEAR_DIST = 12.0f;

float CFG_BEZ_DIST_THRESH = 50.0f;
int   CFG_BEZ_MIN_DELAY = 2;
int   CFG_BEZ_MAX_DELAY = 5;
int   CFG_BEZ_DURATION_MS = 10;
int   CFG_BEZ_STEP_MS = 3;
float CFG_BEZ_DEVIATION = 0.28f;
int   CFG_BEZ_DIRECT_THRESH = 8;
bool  CFG_BEZ_ENABLED = true;

// ============================================================
//  低通滤波器
// ============================================================
class ErrorFilter {
public:
    float filtered = 0.0f;
    bool  initialized = false;
    float update(float raw, float dist) {
        float t = std::min(dist / CFG_FILTER_NEAR_DIST, 1.0f);
        float alpha = CFG_FILTER_ALPHA_NEAR + (1.0f - CFG_FILTER_ALPHA_NEAR) * t;
        if (!initialized) { filtered = raw; initialized = true; }
        else               filtered = alpha * raw + (1.0f - alpha) * filtered;
        return filtered;
    }
    void reset() { initialized = false; filtered = 0.0f; }
};

// ============================================================
//  震荡抑制器
// ============================================================
class OscillationSuppressor {
public:
    float last_out = 0.0f;
    int   flip_cnt = 0;
    int   stable_cnt = 0;

    float update(float output, float dist) {
        if (!CFG_OSC_ENABLED) return output;
        if (dist >= CFG_OSC_NEAR_DIST) {
            flip_cnt = 0; stable_cnt = 0; last_out = output;
            return output;
        }
        bool flip = (last_out != 0.0f) && (output * last_out < 0.0f);
        if (flip) { flip_cnt++; stable_cnt = 0; }
        else {
            stable_cnt++;
            if (stable_cnt >= CFG_OSC_STABLE_FRAMES) {
                flip_cnt = std::max(flip_cnt - 1, 0);
                stable_cnt = 0;
            }
        }
        float result = (flip_cnt >= 1) ? output * CFG_OSC_DECAY : output;
        last_out = result;
        return result;
    }
    void reset() { last_out = 0.0f; flip_cnt = 0; stable_cnt = 0; }
};

// ============================================================
//  亚像素累积器
// ============================================================
class SubPixelAccum {
public:
    float accum = 0.0f;
    int get(float step) {
        accum += step;
        int px = (int)accum;
        accum -= (float)px;
        return px;
    }
    void reset() { accum = 0.0f; }
};

// ============================================================
//  实例化
// ============================================================
static PIDController         pidX(0.20f, 0.018f, 0.002f);
static PIDController         pidY(0.20f, 0.018f, 0.002f);
static ErrorFilter           filterX, filterY;
static OscillationSuppressor oscX, oscY;
static SubPixelAccum         accumX, accumY;

static bool g_pid_initialized = false;

// PID 参数热更新（用重建代替 setGains，PIDController 没有该方法）
static float s_last_kp = -1, s_last_ki = -1, s_last_kd = -1;
void ApplyPIDParams() {
    if (CFG_KP != s_last_kp || CFG_KI != s_last_ki || CFG_KD != s_last_kd) {
        pidX = PIDController(CFG_KP, CFG_KI, CFG_KD);
        pidY = PIDController(CFG_KP, CFG_KI, CFG_KD);
        s_last_kp = CFG_KP; s_last_ki = CFG_KI; s_last_kd = CFG_KD;
        g_pid_initialized = false;
    }
    pidX.setPredictionMaxOrderTaylor(CFG_PRED_ORDER);
    pidY.setPredictionMaxOrderTaylor(CFG_PRED_ORDER);
    pidX.setInitScale(CFG_INIT_SCALE);
    pidY.setInitScale(CFG_INIT_SCALE);
    pidX.setRampTimeMs(CFG_RAMP_MS);
    pidY.setRampTimeMs(CFG_RAMP_MS);
    pidX.setOutputLimits(-CFG_OUT_MAX, CFG_OUT_MAX);
    pidY.setOutputLimits(-CFG_OUT_MAX, CFG_OUT_MAX);
}

static void EnsurePIDInit() {
    if (g_pid_initialized) return;
    ApplyPIDParams();
    pidX.setPredictionWeight(0.0f);
    pidY.setPredictionWeight(0.0f);
    g_pid_initialized = true;
}

// ============================================================
//  计时
// ============================================================
static auto          g_clk_start = std::chrono::steady_clock::now();
static unsigned long g_last_ms = 0;
static unsigned long ms_now() {
    return (unsigned long)std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - g_clk_start).count();
}

// ============================================================
//  贝塞尔
// ============================================================
static void MoveBezier(int total_dx, int total_dy)
{
    if (total_dx == 0 && total_dy == 0) return;
    if ((int)sqrtf((float)(total_dx * total_dx + total_dy * total_dy)) < CFG_BEZ_DIRECT_THRESH) {
        DoMouseMove(total_dx, total_dy);
        return;
    }
    int   steps = std::max(2, CFG_BEZ_DURATION_MS / CFG_BEZ_STEP_MS);
    float r1 = ((rand() % 200 - 100) / 100.0f) * CFG_BEZ_DEVIATION;
    float r2 = ((rand() % 200 - 100) / 100.0f) * CFG_BEZ_DEVIATION;
    float p1x = total_dx * 0.25f - total_dy * r1, p1y = total_dy * 0.25f + total_dx * r1;
    float p2x = total_dx * 0.75f - total_dy * r2, p2y = total_dy * 0.75f + total_dx * r2;
    int lx = 0, ly = 0;
    for (int i = 1; i <= steps; ++i) {
        float t = (float)i / steps, u = 1.f - t;
        float cx = 3 * u * u * t * p1x + 3 * u * t * t * p2x + t * t * t * total_dx;
        float cy = 3 * u * u * t * p1y + 3 * u * t * t * p2y + t * t * t * total_dy;
        int nx = (int)roundf(cx), ny = (int)roundf(cy);
        int sx = nx - lx, sy = ny - ly;
        if (sx || sy) DoMouseMove(sx, sy);
        lx = nx; ly = ny;
        if (i < steps) Sleep(CFG_BEZ_STEP_MS);
    }
    int rx = total_dx - lx, ry = total_dy - ly;
    if (rx || ry) DoMouseMove(rx, ry);
}

// ============================================================
//  SendData
// ============================================================
void SendData(int targetX, int targetY, bool isfire, float sendOffset, int /*boxW*/)
{
    EnsurePIDInit();
    ApplyPIDParams();

    if (targetX < 100 || targetY < 100 || targetX > 3000 || targetY > 2000)
        return;

    float raw_x = (float)(targetX - GetCenterX());
    float raw_y = (float)(targetY - GetCenterY());
    float dist = sqrtf(raw_x * raw_x + raw_y * raw_y);

    if (dist <= CFG_DZ_DIST) {
        pidX.reset(); pidY.reset();
        filterX.reset(); filterY.reset();
        oscX.reset();    oscY.reset();
        accumX.reset();  accumY.reset();
        if (isfire) { DoMouseLeft(1); Sleep(2); DoMouseLeft(0); }
        return;
    }

    unsigned long now = ms_now();
    float dt_ms = (float)(now - g_last_ms);
    if (dt_ms <= 0.f || dt_ms > 200.f) dt_ms = 16.f;
    g_last_ms = now;

    float pred_t = std::clamp(
        (dist - CFG_PRED_ZERO_DIST) / (CFG_PRED_FULL_DIST - CFG_PRED_ZERO_DIST),
        0.0f, 1.0f);
    pidX.setPredictionWeight(CFG_PRED_WEIGHT_X * pred_t);
    pidY.setPredictionWeight(CFG_PRED_WEIGHT_Y * pred_t);

    if (dist < CFG_INTEGRAL_CLEAR_DIST) {
        pidX.resetIntegral(); pidY.resetIntegral();
        accumX.reset();       accumY.reset();
    }

    float filtered_x = filterX.update(raw_x, dist);
    float filtered_y = filterY.update(raw_y, dist);

    float offset_scale = std::min(dist / CFG_OFFSET_RAMP, 1.0f);
    pidX.setAimPos(filtered_x);
    pidY.setAimPos(filtered_y + sendOffset * offset_scale);
    float step_x = pidX.update(0.0f, dt_ms);
    float step_y = pidY.update(0.0f, dt_ms);

    auto getMax = [&](float d) -> int {
        if (d < CFG_NEAR_DIST) return CFG_MAX_NEAR;
        if (d < CFG_MID_DIST)  return CFG_MAX_MID;
        if (d < CFG_FAR_DIST)  return CFG_MAX_FFAR;
        return CFG_MAX_FAR;
        };
    int mx = getMax(dist), my = getMax(dist);

    int accum_x = accumX.get(step_x);
    int accum_y = accumY.get(step_y);
    int clamped_x = std::clamp(accum_x, -mx, mx);
    int clamped_y = std::clamp(accum_y, -my, my);

    int final_x = (int)roundf(oscX.update((float)clamped_x, dist));
    int final_y = (int)roundf(oscY.update((float)clamped_y, dist));

    if (!CFG_BEZ_ENABLED || dist < CFG_BEZ_DIST_THRESH) {
        DoMouseMove(final_x, final_y);
    }
    else {
        int delay = CFG_BEZ_MIN_DELAY + rand() % (CFG_BEZ_MAX_DELAY - CFG_BEZ_MIN_DELAY + 1);
        Sleep(delay);
        MoveBezier(final_x, final_y);
    }

    if (isfire) { DoMouseLeft(1); Sleep(2); DoMouseLeft(0); }
}
