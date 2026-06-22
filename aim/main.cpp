#include "all.h"
#include "ImGuiUI.h"
#include <mmsystem.h>
#pragma comment(lib, "winmm.lib")
#pragma comment(linker, "/ENTRY:mainCRTStartup")

unsigned int xbox_mac = 0;
static std::unique_ptr<Detector> g_detector;
HANDLE hSerial;

struct ImageData {
    std::vector<uint8_t> data;
    int width;
    int height;
};

static bool test = true;
int  open_test = 1;
static ScreenCapture              g_capture;
std::atomic<bool>          g_running{ false };
std::atomic<bool>          g_requestQuit{ false };

// 工作线程"已停在安全点"确认标志
static std::atomic<bool>          g_capParked{ false };
static std::atomic<bool>          g_dispParked{ false };

// UI 请求：开始/继续运行（主控线程消费）
std::atomic<bool>          g_requestResume{ false };
// UI 请求：暂停（主控线程消费，执行全停）
std::atomic<bool>          g_requestPause{ false };
// UI 请求：应用截图区域变更（主控线程消费）
std::atomic<bool>                 g_requestRegionApply{ false };

static std::shared_ptr<ImageData> g_latestFrame{ nullptr };
static std::mutex                 g_frameMutex;
static std::atomic<double>        g_captureFPS{ 0.0 };

// 截图半径（UI 直接改这个 int），改动后通过 g_requestRegionApply 通知主控线程
int                        g_captureRadius = 112;

// ── 兼容旧 UI：保留 g_paused 作为"对外显示状态"。真正的控制在 g_running。
//    UI 仍可读它来显示"已暂停"。主控线程负责维护它。
std::atomic<bool>                 g_paused{ false };
static const int MODEL_INPUT_SIZE = 224;
// ★ 尺寸改为原子，避免多线程无锁读写产生撕裂的中间态
static std::atomic<int>           g_capW{ g_captureRadius * 2 };
static std::atomic<int>           g_capH{ g_captureRadius * 2 };
static int  g_backend = 1;
static const char* g_backendNames[] = { "GDI", "DXGI" };

// ── 性能计时（UI 显示）──────────────────────────────────────
std::atomic<float> g_captureMs{ 0.f };   // 截图耗时 ms
std::atomic<float> g_inferMs{ 0.f };   // 推理耗时 ms
std::atomic<float> g_inferFPS{ 0.f };   // 推理帧率

// ── 控制台日志环形缓冲 ───────────────────────────────────────
#include <mutex>
#include <deque>
std::mutex               g_logMutex;
std::deque<std::string>  g_logBuf;
static constexpr int     LOG_MAX = 300;
void PushLog(const std::string& s) {
    std::lock_guard<std::mutex> lk(g_logMutex);
    g_logBuf.push_back(s);
    if ((int)g_logBuf.size() > LOG_MAX) g_logBuf.pop_front();
}

// ── 自瞄参数 ────────────────────────────────────────────────
int   y_offset = 12;
int   MAX_LOST_FRAMES = 6;
float PREDICT_TIME = 0.007f;
float VELOCITY_ALPHA = 0.35f;
float MAX_VELOCITY = 800.0f;
std::vector<int> g_categories = { 0, 5 };
float conf = 0.25f;
float SAME_TARGET_DIST_SQ = 2500.0f;
bool  g_autoFireEnabled = true;
static bool  fireNow = false;
float g_fireRadius = 3.0f;
float CFG_AIM_OFFSET_PCT = -30.0f;

// ── 自瞄触发键 ───────────────────────────────────────────────
bool g_aimOnLeft = false;   // 左键触发自瞄
bool g_aimOnRight = true;    // 右键触发自瞄

// ── 急停射击（反向按键实现）───────────────────────────────────
// 条件：按下自瞄触发键 AND 有锁定目标。
// 逻辑：玩家按 A → 注入按 D 抵消；按 D → 注入按 A；W↔S 同理。
// 注入后端可切换：关闭 / KMBox反向 / 系统反向。
bool g_counterStrafing = false;

// ── 急停注入后端 ─────────────────────────────────────────────
//   0=关闭  1=KMBox反向(kmNet_enc_keydown/up + HID码)  2=系统反向(SendInput 扫描码)
enum CsBackend { CS_OFF = 0, CS_KMBOX = 1, CS_SYSTEM = 2 };
int g_csBackend = CS_SYSTEM;   // 启动时按 KMBox 是否连接覆盖（见 main）

// ── KMBox 连接状态 ───────────────────────────────────────────
bool g_kmConnected = false;

// Windows VK -> USB HID 键盘 Usage ID（KMBox enc_keydown/keyup 用 HID 扫描码）
//   只需 WASD。返回 0 表示无映射。
static int VkToHid(WORD vk) {
    switch (vk) {
    case 'A': return 0x04;
    case 'D': return 0x07;
    case 'W': return 0x1A;
    case 'S': return 0x16;
    default:  return 0;
    }
}

// Windows VK -> PS/2 Set-1 硬件扫描码（SendInput 用 KEYEVENTF_SCANCODE）
//   很多游戏走 DirectInput/RawInput 只认硬件扫描码，注入 VK 收不到，故用扫描码。
static WORD VkToScan(WORD vk) {
    switch (vk) {
    case 'A': return 0x1E;
    case 'D': return 0x20;
    case 'W': return 0x11;
    case 'S': return 0x1F;
    default:  return (WORD)MapVirtualKeyW(vk, MAPVK_VK_TO_VSC);
    }
}

// ── 急停专用注入：按当前后端发"按下/抬起" ─────────────────────
//   CS_KMBOX : kmNet_enc_keydown/keyup(HID)
//   CS_SYSTEM: SendInput + KEYEVENTF_SCANCODE(硬件扫描码)
static void CsKeyDown(WORD vk) {
    switch (g_csBackend) {
    case CS_KMBOX:
        if (g_kmConnected) { int h = VkToHid(vk); if (h) kmNet_enc_keydown(h); }
        break;
    case CS_SYSTEM: {
        INPUT in = {}; in.type = INPUT_KEYBOARD;
        in.ki.wScan = VkToScan(vk);
        in.ki.dwFlags = KEYEVENTF_SCANCODE;
        SendInput(1, &in, sizeof(INPUT));
        break;
    }
    default: break;
    }
}
static void CsKeyUp(WORD vk) {
    switch (g_csBackend) {
    case CS_KMBOX:
        if (g_kmConnected) { int h = VkToHid(vk); if (h) kmNet_enc_keyup(h); }
        break;
    case CS_SYSTEM: {
        INPUT in = {}; in.type = INPUT_KEYBOARD;
        in.ki.wScan = VkToScan(vk);
        in.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
        SendInput(1, &in, sizeof(INPUT));
        break;
    }
    default: break;
    }
}

// 供 UI 调用（非 static）：切换后端时由 DisplayLoop 自行释放注入键，
// 这里仅作占位/兼容（无残留硬件状态需要清理）。
void UIClearKeyMasks() {}

// ── Raw Input 后台键盘检测（GetAsyncKeyState在游戏有焦点时可能失效）──
std::atomic<bool> g_rawA{ false }, g_rawD{ false }, g_rawW{ false }, g_rawS{ false };

static HWND  g_rawHwnd = nullptr;
static LRESULT CALLBACK RawWndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
    if (m == WM_INPUT) {
        UINT sz = 0;
        GetRawInputData((HRAWINPUT)l, RID_INPUT, nullptr, &sz, sizeof(RAWINPUTHEADER));
        std::vector<BYTE> buf(sz);
        if (GetRawInputData((HRAWINPUT)l, RID_INPUT, buf.data(), &sz, sizeof(RAWINPUTHEADER)) == sz) {
            auto* ri = (RAWINPUT*)buf.data();
            if (ri->header.dwType == RIM_TYPEKEYBOARD) {
                WORD  vk = ri->data.keyboard.VKey;
                bool  down = !(ri->data.keyboard.Flags & RI_KEY_BREAK);
                if (vk == 'A') g_rawA = down;
                if (vk == 'D') g_rawD = down;
                if (vk == 'W') g_rawW = down;
                if (vk == 'S') g_rawS = down;
            }
        }
        return 0;
    }
    return DefWindowProcW(h, m, w, l);
}
static std::thread g_rawThread;
static void StartRawInput() {
    g_rawThread = std::thread([]() {
        WNDCLASSEXW wc{ sizeof(wc) };
        wc.lpfnWndProc = RawWndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"RawKbdWnd";
        RegisterClassExW(&wc);
        g_rawHwnd = CreateWindowExW(0, L"RawKbdWnd", L"", 0,
            0, 0, 0, 0, HWND_MESSAGE, nullptr, wc.hInstance, nullptr);
        RAWINPUTDEVICE rid{ 0x01, 0x06, RIDEV_INPUTSINK, g_rawHwnd };
        RegisterRawInputDevices(&rid, 1, sizeof(rid));
        MSG msg;
        while (GetMessageW(&msg, g_rawHwnd, 0, 0) > 0) {
            TranslateMessage(&msg); DispatchMessageW(&msg);
        }
        DestroyWindow(g_rawHwnd);
        UnregisterClassW(L"RawKbdWnd", GetModuleHandleW(nullptr));
        });
}
static void StopRawInput() {
    if (g_rawHwnd) PostMessageW(g_rawHwnd, WM_QUIT, 0, 0);
    if (g_rawThread.joinable()) g_rawThread.join();
}

// ── 模型/引擎 ────────────────────────────────────────────────
int model_way = 0;
int inference_engine_mode = 1;   // 0=TensorRT(.engine) 1=ONNXRuntime(.onnx) —— 默认 ONNX
int  g_onnxDevice = 1;               // 0=Auto 1=DirectML(MDL) 2=CUDA 3=CPU —— 默认 DirectML(MDL)
bool open_findcolor = false;
int  g_modelInputSize = 224;
char g_modelPath[512] = "";
char g_kmIP[64] = "192.168.2.188";
char g_kmPort[16] = "1000";
char g_kmMAC[32] = "25ABDBB2";

// ── 屏幕分辨率（0=自动读取系统分辨率，可手动覆盖）─────────────
int   g_screenW = 0;          // 0=自动读取
int   g_screenH = 0;          // 0=自动读取
bool  g_useManualRes = false; // true=使用手动输入的分辨率

int GetEffectiveScreenW() {
    if (g_useManualRes && g_screenW > 0) return g_screenW;
    return GetSystemMetrics(SM_CXSCREEN);
}
int GetEffectiveScreenH() {
    if (g_useManualRes && g_screenH > 0) return g_screenH;
    return GetSystemMetrics(SM_CYSCREEN);
}

// ── 找色参数 ─────────────────────────────────────────────────
float fc_h1lo = 0, fc_h1hi = 3, fc_s1lo = 100, fc_s1hi = 255, fc_v1lo = 120, fc_v1hi = 255;
float fc_h2lo = 177, fc_h2hi = 180, fc_s2lo = 100, fc_s2hi = 255, fc_v2lo = 100, fc_v2hi = 255;
int   g_fcRadius = 40;        // 找色区域半径（默认40 → 屏幕中心80×80像素）
int   fc_roi_x = -40, fc_roi_y = -40, fc_roi_w = 80, fc_roi_h = 80;
float fc_aim_scale = 0.7f;
int   fc_min_area = 2;

// ═══════════════════════════════════════════════════════════════════
//  握手工具：主控线程用它确定性地"全停 → 安全操作 → 全启"
// ═══════════════════════════════════════════════════════════════════

// 主控线程调用：要求全停，并阻塞直到两个工作线程都确认 parked。
// 返回后，CaptureLoop 不在 Capture()、DisplayLoop 不在用帧/Inference()。
static void StopAndWaitParked()
{
    g_running.store(false, std::memory_order_release);
    g_paused.store(true, std::memory_order_release);
    // 先清零 parked，强制等待"本次"停止后由 worker 重新置位，
    // 避免上一次放行残留的 true 让等待瞬间通过（防快速连点竞态）。
    g_capParked.store(false, std::memory_order_release);
    g_dispParked.store(false, std::memory_order_release);
    // 自旋等待两线程都停在安全点。worker 每 ~2ms 检查一次，所以这里很快就绪。
    while (!g_capParked.load(std::memory_order_acquire) ||
        !g_dispParked.load(std::memory_order_acquire))
    {
        if (g_requestQuit.load(std::memory_order_acquire)) return; // 退出途中不再等
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// CaptureLoop 内部：（重新）初始化截图后端到当前 g_capW/g_capH。
// 只在 parked（全停）状态下被调用，所以读取尺寸 100% 安全、无撕裂。
static bool CaptureInitToCurrentSize()
{
    int w = g_capW.load(std::memory_order_acquire);
    int h = g_capH.load(std::memory_order_acquire);
    // ★ 防御：尺寸非法直接失败，避免 Init/分配出现负数或 0
    if (w <= 0 || h <= 0) {
        PushLog("[截图] Init 失败(尺寸非法)");
        return false;
    }
    int sw = GetEffectiveScreenW();
    int sh = GetEffectiveScreenH();
    int tx = (sw - w) / 2, ty = (sh - h) / 2;

    g_capture.Reset();
    g_backend = 1;
    if (!g_capture.Init(w, h, g_backend)) {
        g_backend = 0;
        if (!g_capture.Init(w, h, g_backend)) {
            PushLog("[截图] Init 失败(DXGI+GDI均失败)");
            return false;
        }
    }
    g_capture.SetRegion(tx, ty, w, h);

    // 丢弃任何旧尺寸残帧，DisplayLoop 醒来后只会拿到新尺寸帧
    { std::lock_guard<std::mutex> lk(g_frameMutex); g_latestFrame.reset(); }

    char buf[128];
    snprintf(buf, sizeof(buf), "[截图] 初始化 %dx%d 区域(%d,%d) 后端:%s",
        w, h, tx, ty, g_backendNames[g_backend]);
    PushLog(buf);
    return true;
}

// ========================
// 采集线程
// ========================
void CaptureLoop()
{
    CoInitializeEx(nullptr, COINIT_MULTITHREADED);

    int   frameCount = 0;
    auto  lastTime = std::chrono::high_resolution_clock::now();
    bool  inited = false;      // 截图后端是否已初始化（首次放行时初始化）

    while (!g_requestQuit.load()) {
        // ── 未运行：停在安全点，宣告 parked，绝不触碰截图资源 ──────────
        if (!g_running.load(std::memory_order_acquire)) {
            g_capParked.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // ── 刚被放行（parked→active）：在本线程内安全(重)初始化 DXGI ────
        if (g_capParked.exchange(false, std::memory_order_acq_rel)) {
            // 此刻主控线程保证：g_running 已设 true 且尺寸已写好；
            // 我们在自己的线程里初始化（DXGI 要求同线程），完成后开始采集。
            // ★ 修复：初始化失败时绝不能置 inited=true，否则后续会对
            //    未初始化/已 Reset 的 g_capture 调 Capture()，导致未定义行为。
            inited = CaptureInitToCurrentSize();
            frameCount = 0;
            lastTime = std::chrono::high_resolution_clock::now();
            if (!inited) {
                // 初始化失败：回到暂停安全点，等待下一次放行重试，避免空转崩溃。
                g_running.store(false, std::memory_order_release);
                g_paused.store(true, std::memory_order_release);
                g_capParked.store(true, std::memory_order_release);
                PushLog("[截图] 初始化失败，已暂停采集");
                continue;
            }
        }

        if (!inited) {  // 理论上不会发生，保险起见
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        // ★ bufferSize 每次从当前 g_capW/g_capH 快照计算，尺寸变了也不会越界
        int capW = g_capW.load(std::memory_order_acquire);
        int capH = g_capH.load(std::memory_order_acquire);
        size_t bufferSize = (size_t)capW * capH * 3;

        auto frame = std::make_shared<ImageData>();
        frame->width = capW;
        frame->height = capH;
        frame->data.resize(bufferSize);

        // ── 截图计时 ─────────────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();
        bool captured = g_capture.Capture(frame->data.data(), bufferSize);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (!captured) continue;

        float capMs = std::chrono::duration<float, std::milli>(t1 - t0).count();
        g_captureMs.store(capMs);

        frame->width = g_capture.GetWidth();
        frame->height = g_capture.GetHeight();
        { std::lock_guard<std::mutex> lk(g_frameMutex); g_latestFrame = std::move(frame); }

        frameCount++;
        auto   now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastTime).count();
        if (elapsed >= 1.0) {
            g_captureFPS = frameCount / elapsed;
            frameCount = 0;
            lastTime = now;
        }
    }
    // 真正退出才 Reset
    g_capture.Reset();
    CoUninitialize();
}

// ========================
// 显示 + 自瞄主循环
// ========================
void DisplayLoop()
{
    const std::string windowName = "Screen Capture Demo";
    std::shared_ptr<ImageData> prevFrame;
    cv::Mat display;

    // ★ 几何量不再用启动时的 const 快照——截图区域可在运行时改变。
    //   下列变量改为每帧根据当前 frame 尺寸重新计算（见循环内）。
    int screenW = GetEffectiveScreenW();
    int screenH = GetEffectiveScreenH();

    bool      hasLockedTarget = false;
    int       lostFrameCount = 0;
    int       lockedCX = 0, lockedCY = 0;
    Detection lockedTarget;
    float     vx = 0.f, vy = 0.f;
    bool      velocityReady = false;

    // ── 急停持键状态（记录我们当前注入按住了哪些键）──────────────
    bool cs_holdD = false, cs_holdA = false;
    bool cs_holdS = false, cs_holdW = false;

    // 释放所有急停注入键的 lambda（停止/暂停/松键/丢失目标时调用）
    auto csReleaseAll = [&]() {
        if (cs_holdD) { CsKeyUp('D'); cs_holdD = false; }
        if (cs_holdA) { CsKeyUp('A'); cs_holdA = false; }
        if (cs_holdS) { CsKeyUp('S'); cs_holdS = false; }
        if (cs_holdW) { CsKeyUp('W'); cs_holdW = false; }
        };

    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    auto lastFpsTime = std::chrono::high_resolution_clock::now();

    // 推理帧计数
    int   inferCount = 0;
    auto  inferFpsTimer = std::chrono::high_resolution_clock::now();

    bool wasRunning = false;  // 上一帧是否在运行（用于检测恢复时机）
    int  prevCsBackend = g_csBackend;  // 检测急停后端切换，安全释放旧后端注入键

    // 用指定后端释放某键（不依赖全局 g_csBackend，避免与 UI 线程竞态）
    auto csKeyUpVia = [&](int backend, WORD vk) {
        if (backend == CS_KMBOX) {
            if (g_kmConnected) { int h = VkToHid(vk); if (h) kmNet_enc_keyup(h); }
        }
        else if (backend == CS_SYSTEM) {
            INPUT in = {}; in.type = INPUT_KEYBOARD;
            in.ki.wScan = VkToScan(vk);
            in.ki.dwFlags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP;
            SendInput(1, &in, sizeof(INPUT));
        }
        };

    while (!g_requestQuit.load())
    {
        // ── 急停后端切换：用"旧后端"释放仍按住的注入键，避免卡键 ──
        int curCsBackend = g_csBackend;  // 本帧快照（UI 线程可能随时改）
        if (curCsBackend != prevCsBackend) {
            if (cs_holdD) { csKeyUpVia(prevCsBackend, 'D'); cs_holdD = false; }
            if (cs_holdA) { csKeyUpVia(prevCsBackend, 'A'); cs_holdA = false; }
            if (cs_holdS) { csKeyUpVia(prevCsBackend, 'S'); cs_holdS = false; }
            if (cs_holdW) { csKeyUpVia(prevCsBackend, 'W'); cs_holdW = false; }
            prevCsBackend = curCsBackend;
        }
        // ── 未运行：停在安全点，宣告 parked，绝不用旧帧/不调用 Inference ──
        if (!g_running.load(std::memory_order_acquire)) {
            if (wasRunning) {
                // 刚从运行变为停止：释放急停键，彻底重置跟踪状态
                csReleaseAll();
                hasLockedTarget = false; lostFrameCount = 0;
                vx = vy = 0.f; velocityReady = false;
                cs_holdD = cs_holdA = cs_holdS = cs_holdW = false;
                prevFrame.reset();   // ★ 丢弃旧尺寸帧引用，恢复后只认新帧
            }
            wasRunning = false;
            g_dispParked.store(true, std::memory_order_release);
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        // 被放行：清掉 parked，重新进入工作
        if (!wasRunning) {
            g_dispParked.store(false, std::memory_order_release);
            prevFrame.reset();
            lastFrameTime = std::chrono::high_resolution_clock::now();
        }
        wasRunning = true;

        auto  now = std::chrono::high_resolution_clock::now();
        float dt = std::chrono::duration<float>(now - lastFrameTime).count();
        lastFrameTime = now;
        if (dt <= 0.f || dt > 0.1f) dt = 0.016f;

        bool isPPressed = (GetAsyncKeyState('P') & 0x8000) != 0;
        if (isPPressed) { csReleaseAll(); g_requestQuit = true; continue; }

        std::shared_ptr<ImageData> frame;
        { std::lock_guard<std::mutex> lk(g_frameMutex); frame = g_latestFrame; }

        if (!frame || frame == prevFrame) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        prevFrame = frame;
        if (frame->data.empty()) continue;

        // ★ 防御：若 data 大小与声明的宽高不一致（极端竞态下的撕裂帧），跳过本帧
        if ((size_t)frame->width * frame->height * 3 != frame->data.size()
            || frame->width <= 0 || frame->height <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        // ★ 几何量每帧根据当前 frame 尺寸重新推导（截图区域可运行时改变）
        const int captureW = frame->width;
        const int captureH = frame->height;
        const int screenCenterX = captureW / 2;
        const int screenCenterY = captureH / 2;
        const int captureOffsetX = (screenW - captureW) / 2;
        const int captureOffsetY = (screenH - captureH) / 2;

        cv::Mat frameMat(frame->height, frame->width, CV_8UC3,
            const_cast<uint8_t*>(frame->data.data()));

        if (display.rows != frame->height || display.cols != frame->width)
            display.create(frame->height, frame->width, CV_8UC3);
        frameMat.copyTo(display);

        cv::circle(display, cv::Point(screenCenterX, screenCenterY),
            4, cv::Scalar(255, 255, 255), -1);

        if (g_detector)
        {
            // ── 推理计时 ─────────────────────────────────────
            auto ti0 = std::chrono::high_resolution_clock::now();
            std::vector<Detection> results = g_detector->Inference(frameMat);
            auto ti1 = std::chrono::high_resolution_clock::now();
            float inferMs = std::chrono::duration<float, std::milli>(ti1 - ti0).count();
            g_inferMs.store(inferMs);

            // 推理 FPS 统计
            inferCount++;
            double iElapsed = std::chrono::duration<double>(ti1 - inferFpsTimer).count();
            if (iElapsed >= 1.0) {
                g_inferFPS.store((float)(inferCount / iElapsed));
                inferCount = 0;
                inferFpsTimer = ti1;
            }

            // ── 找色 ─────────────────────────────────────────
            // 从半径同步 ROI 偏移量（保证 UI 只改 g_fcRadius 即可）
            fc_roi_x = -g_fcRadius;
            fc_roi_y = -g_fcRadius;
            fc_roi_w = g_fcRadius * 2;
            fc_roi_h = g_fcRadius * 2;
            if (open_findcolor) {
                cv::Rect roi(screenCenterX + fc_roi_x, screenCenterY + fc_roi_y,
                    fc_roi_w, fc_roi_h);
                roi &= cv::Rect(0, 0, frameMat.cols, frameMat.rows);
                cv::Mat roiMat = frameMat(roi);
                FindColor(roiMat, (float)screenW, (float)screenH);
                // ★ 红色矩形框出找色区域
                cv::rectangle(display, roi, cv::Scalar(0, 0, 255), 1);
                cv::putText(display, "FC_ROI",
                    cv::Point(roi.x, std::max(roi.y - 4, 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 0, 255), 1);
            }

            // ── 优先级目标搜索 ───────────────────────────────
            bool      foundCandidate = false;
            Detection candidateTarget;
            float     candidateDistSq = FLT_MAX;
            int       candCX = 0, candCY = 0;

            auto tryFind = [&](auto filterFn) {
                for (const auto& det : results) {
                    if (!filterFn(det.classId)) continue;
                    if (det.box.width <= 0 || det.box.height <= 0) continue;
                    if (det.conf < conf || std::isnan(det.conf)) continue;
                    int   cx = det.box.x + det.box.width / 2;
                    int   cy = det.box.y + det.box.height / 2;
                    float dSq = (float)((cx - screenCenterX) * (cx - screenCenterX) +
                        (cy - screenCenterY) * (cy - screenCenterY));
                    if (dSq < candidateDistSq) {
                        candidateDistSq = dSq;
                        candidateTarget = det;
                        candCX = cx; candCY = cy;
                        foundCandidate = true;
                    }
                }
                };

            for (int cat : g_categories) {
                tryFind([cat](int id) { return id == cat; });
                if (foundCandidate) break;
            }

            // ── 目标跟踪 ─────────────────────────────────────
            if (foundCandidate) {
                float sameDistSq = hasLockedTarget
                    ? (float)((candCX - lockedCX) * (candCX - lockedCX) +
                        (candCY - lockedCY) * (candCY - lockedCY))
                    : FLT_MAX;
                bool isSameTarget = (sameDistSq < SAME_TARGET_DIST_SQ);

                if (!hasLockedTarget) {
                    hasLockedTarget = true; lostFrameCount = 0;
                    vx = vy = 0.f; velocityReady = false;
                }
                else if (isSameTarget) {
                    float rawVx = (candCX - lockedCX) / dt;
                    float rawVy = (candCY - lockedCY) / dt;
                    if (!velocityReady) { vx = rawVx; vy = rawVy; velocityReady = true; }
                    else {
                        vx = VELOCITY_ALPHA * rawVx + (1.f - VELOCITY_ALPHA) * vx;
                        vy = VELOCITY_ALPHA * rawVy + (1.f - VELOCITY_ALPHA) * vy;
                    }
                    vx = std::clamp(vx, -MAX_VELOCITY, MAX_VELOCITY);
                    vy = std::clamp(vy, -MAX_VELOCITY, MAX_VELOCITY);
                    lostFrameCount = 0;
                }
                else {
                    if (lostFrameCount >= MAX_LOST_FRAMES) {
                        hasLockedTarget = true; lostFrameCount = 0;
                        vx = vy = 0.f; velocityReady = false;
                    }
                    else { lostFrameCount++; goto draw_and_send; }
                }
                lockedTarget = candidateTarget;
                lockedCX = candCX; lockedCY = candCY;
            }
            else {
                if (hasLockedTarget) {
                    lostFrameCount++;
                    if (lostFrameCount > MAX_LOST_FRAMES) {
                        hasLockedTarget = false; lostFrameCount = 0;
                        vx = vy = 0.f; velocityReady = false;
                        csReleaseAll(); // 目标丢失：释放急停键
                    }
                }
            }

        draw_and_send:
            if (hasLockedTarget && lostFrameCount == 0)
            {
                int localCX = std::clamp(lockedCX, 0, captureW - 1);
                int localCY = std::clamp(lockedCY, 0, captureH - 1);
                int boxH = lockedTarget.box.height;
                int boxW = lockedTarget.box.width;
                int offsetPx = (int)((std::clamp(CFG_AIM_OFFSET_PCT, -100.f, 100.f) / 100.f) * (boxH * 0.5f));
                localCY = std::clamp(localCY + offsetPx, 0, captureH - 1);

                int predX = localCX, predY = localCY;
                if (velocityReady) {
                    predX = std::clamp((int)(localCX + vx * PREDICT_TIME), 0, captureW - 1);
                    predY = std::clamp((int)(localCY + vy * PREDICT_TIME), 0, captureH - 1);
                }

                int absoluteX = predX + captureOffsetX;
                int absoluteY = predY + captureOffsetY;

                // ── 自瞄触发判断：系统检测 OR KMBox 满足其一 ──
                bool _sysL = (GetAsyncKeyState(VK_LBUTTON) & 0x8000) != 0;
                bool _sysR = (GetAsyncKeyState(VK_RBUTTON) & 0x8000) != 0;
                bool _kmL = g_kmConnected && kmNet_monitor_mouse_left() != 0;
                bool _kmR = g_kmConnected && kmNet_monitor_mouse_right() != 0;
                bool _trigL = g_aimOnLeft && (_sysL || _kmL);
                bool _trigR = g_aimOnRight && (_sysR || _kmR);
                bool _doAim = _trigL || _trigR;

                if (_doAim) {
                    // 自动开火
                    fireNow = false;
                    if (g_autoFireEnabled) {
                        float _dc = sqrtf((float)(
                            (predX - screenCenterX) * (predX - screenCenterX) +
                            (predY - screenCenterY) * (predY - screenCenterY)));
                        fireNow = (_dc < g_fireRadius);
                    }
                    if (_trigL) absoluteY += y_offset;
                    SendData(absoluteX, absoluteY, fireNow, 0, boxW);

                    // ── 急停射击（反向按键实现）─────────────────
                    // 条件：按下自瞄触发键 AND 有锁定目标（已在此分支内）。
                    // 玩家按 A → 注入按 D 抵消；按 D → 注入按 A；W↔S 同理。
                    if (g_counterStrafing && g_csBackend != CS_OFF) {
                        // ★ 排除自己注入的按键，防止反馈循环：
                        //   cs_holdD=true 时我们注入了 D，不能把这个 D 算作玩家按的 D，
                        //   否则：注入D → 检测到D → 以为玩家按D → 释放D → 再注入 → 震荡。
                        bool _kA = (g_rawA.load() || (GetAsyncKeyState('A') & 0x8000) != 0) && !cs_holdA;
                        bool _kD = (g_rawD.load() || (GetAsyncKeyState('D') & 0x8000) != 0) && !cs_holdD;
                        bool _kW = (g_rawW.load() || (GetAsyncKeyState('W') & 0x8000) != 0) && !cs_holdW;
                        bool _kS = (g_rawS.load() || (GetAsyncKeyState('S') & 0x8000) != 0) && !cs_holdS;

                        // A按住且D没按 → 持续按D
                        bool wD = _kA && !_kD;
                        if (wD && !cs_holdD) { CsKeyDown('D'); cs_holdD = true; PushLog("[急停] A→按D"); }
                        if (!wD && cs_holdD) { CsKeyUp('D');   cs_holdD = false; }
                        // D按住且A没按 → 持续按A
                        bool wA = _kD && !_kA;
                        if (wA && !cs_holdA) { CsKeyDown('A'); cs_holdA = true; PushLog("[急停] D→按A"); }
                        if (!wA && cs_holdA) { CsKeyUp('A');   cs_holdA = false; }
                        // W按住且S没按 → 持续按S
                        bool wS = _kW && !_kS;
                        if (wS && !cs_holdS) { CsKeyDown('S'); cs_holdS = true; }
                        if (!wS && cs_holdS) { CsKeyUp('S');   cs_holdS = false; }
                        // S按住且W没按 → 持续按W
                        bool wW = _kS && !_kW;
                        if (wW && !cs_holdW) { CsKeyDown('W'); cs_holdW = true; }
                        if (!wW && cs_holdW) { CsKeyUp('W');   cs_holdW = false; }
                    }
                }
                else {
                    // 自瞄键松开：立即释放急停注入键
                    csReleaseAll();
                }

                // ── 绘制 ─────────────────────────────────────
                cv::rectangle(display, lockedTarget.box, cv::Scalar(0, 255, 0), 2);
                cv::circle(display, cv::Point(localCX, localCY), 5, cv::Scalar(0, 0, 255), -1);
                cv::circle(display, cv::Point(predX, predY), 4, cv::Scalar(0, 255, 255), -1);
                cv::line(display, cv::Point(screenCenterX, screenCenterY),
                    cv::Point(predX, predY), cv::Scalar(255, 0, 0), 1);
                cv::putText(display,
                    "ID:" + std::to_string(lockedTarget.classId) +
                    " Conf:" + cv::format("%.2f", lockedTarget.conf) +
                    " Vx:" + cv::format("%.0f", vx) + " Vy:" + cv::format("%.0f", vy),
                    cv::Point(lockedTarget.box.x, std::max(lockedTarget.box.y - 10, 10)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 255, 0), 1);

                // 推遥测
                float dist = sqrtf((float)(
                    (predX - screenCenterX) * (predX - screenCenterX) +
                    (predY - screenCenterY) * (predY - screenCenterY)));
                ImGuiUI::Get().PushTelemetry(
                    g_inferFPS.load(), true,
                    lockedTarget.conf, dist, vx, vy,
                    lockedTarget.classId);

                // 日志（降频30帧）
                static int _lt = 0;
                if (++_lt >= 30) {
                    _lt = 0;
                    char lb[128];
                    snprintf(lb, sizeof(lb),
                        "[锁定] cls=%d conf=%.2f dist=%.0f infer=%.1fms cap=%.1fms",
                        lockedTarget.classId, lockedTarget.conf, dist,
                        g_inferMs.load(), g_captureMs.load());
                    PushLog(lb);
                }
            }
        }

        if (!hasLockedTarget)
            ImGuiUI::Get().PushTelemetry(g_inferFPS.load(), false, 0, 0, 0, 0, -1);

        // ── FPS & 显示 ───────────────────────────────────────
        if (test) {
            auto  nowFps = std::chrono::high_resolution_clock::now();
            float frameMs = std::chrono::duration<float, std::milli>(
                nowFps - lastFpsTime).count();
            float fps = (frameMs > 0.f) ? (1000.f / frameMs) : 0.f;  // ★ 防除零/inf
            lastFpsTime = nowFps;
            cv::putText(display, "FPS:" + std::to_string((int)fps),
                cv::Point(10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 255), 2);
            if (open_test) cv::imshow(windowName, display);
        }
        if (open_test) {
            MSG msg;
            while (PeekMessage(&msg, nullptr, 0, 0, PM_REMOVE)) {
                if (msg.message == WM_QUIT) { g_requestQuit = true; break; }
                TranslateMessage(&msg); DispatchMessage(&msg);
            }
            if (g_requestQuit.load()) break;
            if (GetAsyncKeyState('P') & 0x8000) { csReleaseAll(); g_requestQuit = true; break; }
        }
        else {
            cv::destroyAllWindows();
        }
    }
    csReleaseAll();
    cv::destroyAllWindows();
}

// ========================
// 主函数
// ========================
int main()
{
    SetConsoleOutputCP(65001);
    SetConsoleCP(65001);

    int ret = kmNet_init(g_kmIP, g_kmPort, g_kmMAC);
    if (ret != 0) {
        printf("[KMBox] 连接失败(ret=%d)，降级 SendInput\n", ret);
        g_kmConnected = false;
        PushLog("[KMBox] 连接失败，使用系统 SendInput");
    }
    else {
        printf("[KMBox] 连接成功！\n");
        g_kmConnected = true;
        kmNet_monitor(1);
        PushLog("[KMBox] 连接成功");
    }

    // ── 急停后端默认值：有 KMBox 连接默认 KMBox反向，否则系统反向。
    //    仅在启动时设定一次；用户之后可在 UI 自由切换，不会被覆盖。
    g_csBackend = g_kmConnected ? CS_KMBOX : CS_SYSTEM;
    PushLog(g_kmConnected ? "[急停] 默认: KMBox反向" : "[急停] 默认: 系统反向");

    std::string path(g_modelPath);
    SetMoveMethod(g_kmConnected ? 0 : 1);

    g_capW.store(g_captureRadius * 2, std::memory_order_release);
    g_capH.store(g_captureRadius * 2, std::memory_order_release);

    {
        char buf[128];
        snprintf(buf, sizeof(buf), "[系统] 截图%dx%d 模型输入%dx%d",
            g_capW.load(), g_capH.load(), g_modelInputSize, g_modelInputSize);
        PushLog(buf);
    }

    // ── 启动时不强制加载模型 ────────────────────
    //   模型路径默认为空，避免因路径无效而报错退出。
    //   待用户在 UI 设置好路径并点击"开始"后，由主控线程重载引擎。
    if (path.empty()) {
        PushLog("[引擎] 未设置模型路径，启动后请在界面填写并点击开始");
    }
    else {
        try {
            g_detector = std::make_unique<Detector>(
                path, model_way,
                static_cast<InferenceEngine>(inference_engine_mode));
            g_detector->SetInputSize(g_modelInputSize);
            g_detector->SetONNXDevice(g_onnxDevice);
            char buf[128];
            snprintf(buf, sizeof(buf), "[引擎] 加载成功 %dx%d", g_modelInputSize, g_modelInputSize);
            PushLog(buf);
        }
        catch (const std::exception& e) {
            // 不再退出：仅记录失败，保持暂停，等待用户修正路径后重试。
            char buf[256];
            snprintf(buf, sizeof(buf), "[引擎] 加载失败: %s（保持暂停，请修正路径后点击开始）", e.what());
            PushLog(buf);
            fprintf(stderr, "%s\n", buf);
            g_detector.reset();
        }
    }

    timeBeginPeriod(1);
    StartRawInput();
    ImGuiUI::Get().Start();

    // CaptureLoop 和 DisplayLoop 是常驻线程；g_running=false 时它们 park 在安全点。
    // 启动时 g_running=false，两线程先停在 parked。
    std::thread captureThread(CaptureLoop);
    std::thread displayThread([] { DisplayLoop(); });

    // 设定初始尺寸（此时两线程已 parked，安全）
    g_capW.store(g_captureRadius * 2, std::memory_order_release);
    g_capH.store(g_captureRadius * 2, std::memory_order_release);

    // ── 启动后自动放行一次，进入运行 ─────────────────────────────
    // 让两线程从 parked→active；CaptureLoop 会在自己的线程内首次 Init DXGI。
    // 启动时默认暂停：不自动放行，保持两线程 parked。
    // 等用户设置好模型路径并点击"开始"，由主控循环 (C) 分支重载引擎后再放行。
    g_running.store(false, std::memory_order_release);
    g_paused.store(true, std::memory_order_release);
    g_capParked.store(true, std::memory_order_release);   // 标记已停在安全点
    g_dispParked.store(true, std::memory_order_release);
    PushLog("[系统] 已暂停（启动默认）。请设置模型路径后点击开始");

    // ════════════════════════════════════════════════════════════════
    //  主控循环：所有"会改变共享资源/尺寸/引擎"的操作都遵循同一铁律——
    //    1) StopAndWaitParked()  → 两线程确定性全停（不是 sleep 猜测）
    //    2) 安全地改尺寸 / reload 引擎 / reconnect
    //    3) 重新放行（g_running=true），CaptureLoop 在自己线程 reinit DXGI
    //  因此"改截图区域"再也不会与正在进行的 Capture/Inference 撞车。
    // ════════════════════════════════════════════════════════════════
    while (!g_requestQuit.load())
    {
        // ── (A) 暂停请求：全停，停在 parked，不动任何资源 ─────────────
        if (g_requestPause.exchange(false)) {
            if (g_running.load()) {
                StopAndWaitParked();              // 两线程确定性停下
                PushLog("[系统] 已暂停（全部停止）");
            }
            continue;
        }

        // ── (B) 截图区域变更：全停 → 改尺寸 → 放行（让 Capture 重 Init）──
        if (g_requestRegionApply.exchange(false)) {
            bool wasRunning = g_running.load();

            StopAndWaitParked();                  // ★ 确定性全停，不再 sleep 猜
            if (g_requestQuit.load()) break;

            // 此刻两线程都 parked：改尺寸 100% 安全
            g_capW.store(g_captureRadius * 2, std::memory_order_release);
            g_capH.store(g_captureRadius * 2, std::memory_order_release);
            { std::lock_guard<std::mutex> lk(g_frameMutex); g_latestFrame.reset(); }
            {
                char buf[64];
                snprintf(buf, sizeof(buf), "[截图] 区域变更 %dx%d（已全停后应用）",
                    g_capW.load(), g_capH.load());
                PushLog(buf);
            }

            // 只有原本在运行才恢复；原本暂停则保持暂停
            if (wasRunning) {
                g_capParked.store(true, std::memory_order_release);  // 触发 Capture 重 Init
                g_paused.store(false, std::memory_order_release);
                g_running.store(true, std::memory_order_release);
                PushLog("[系统] 区域已生效，继续运行");
            }
            continue;
        }

        // ── (C) 开始/继续请求：全停 → 重载引擎/KMBox → 放行 ──────────
        if (g_requestResume.exchange(false)) {
            PushLog("[系统] 收到开始/继续，重载配置…");

            StopAndWaitParked();                  // ★ 确定性全停
            if (g_requestQuit.load()) break;

            // 路径为空：不放行，保持暂停，提示用户设置路径。
            if (std::string(g_modelPath).empty()) {
                PushLog("[引擎] 模型路径为空，无法开始。请先设置路径");
                g_capParked.store(true, std::memory_order_release);
                g_dispParked.store(true, std::memory_order_release);
                g_paused.store(true, std::memory_order_release);
                g_running.store(false, std::memory_order_release);
                continue;
            }

            // 重载推理引擎（两线程已 parked，g_detector 无人使用）
            bool engineOk = false;
            try {
                g_detector.reset();
                g_detector = std::make_unique<Detector>(
                    std::string(g_modelPath), model_way,
                    static_cast<InferenceEngine>(inference_engine_mode));
                g_detector->SetInputSize(g_modelInputSize);
                g_detector->SetONNXDevice(g_onnxDevice);
                engineOk = true;
                char buf[128];
                snprintf(buf, sizeof(buf), "[引擎] 重载成功 %dx%d", g_modelInputSize, g_modelInputSize);
                PushLog(buf);
            }
            catch (const std::exception& e) {
                char buf[256];
                snprintf(buf, sizeof(buf), "[引擎] 重载失败: %s", e.what());
                PushLog(buf);
                g_detector.reset();
            }

            // 引擎加载失败：保持暂停，不放行，避免无检测器运行。
            if (!engineOk) {
                PushLog("[系统] 引擎未就绪，保持暂停。请修正路径后重试");
                g_capParked.store(true, std::memory_order_release);
                g_dispParked.store(true, std::memory_order_release);
                g_paused.store(true, std::memory_order_release);
                g_running.store(false, std::memory_order_release);
                continue;
            }

            // 同步尺寸（CaptureLoop 放行后会据此重 Init DXGI）
            g_capW.store(g_captureRadius * 2, std::memory_order_release);
            g_capH.store(g_captureRadius * 2, std::memory_order_release);
            { std::lock_guard<std::mutex> lk(g_frameMutex); g_latestFrame.reset(); }

            // KMBox 重连
            {
                int r = kmNet_init(g_kmIP, g_kmPort, g_kmMAC);
                if (r == 0) {
                    g_kmConnected = true; kmNet_monitor(1); SetMoveMethod(0);
                    PushLog("[KMBox] 重连成功");
                }
                else {
                    g_kmConnected = false; SetMoveMethod(1);
                    char buf[64];
                    snprintf(buf, sizeof(buf), "[KMBox] 重连失败(ret=%d)", r);
                    PushLog(buf);
                }
            }

            // 放行：CaptureLoop 在自己线程内 reinit DXGI（同线程要求）
            g_capParked.store(true, std::memory_order_release);  // 触发 Capture 重 Init 分支
            g_paused.store(false, std::memory_order_release);
            g_running.store(true, std::memory_order_release);
            PushLog("[系统] 运行中");
            continue;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }

    // ── 彻底退出：先全停，再 join ─────────────────────────────────
    g_running.store(false, std::memory_order_release);
    g_paused.store(true, std::memory_order_release);
    StopRawInput();
    if (captureThread.joinable()) captureThread.join();
    if (displayThread.joinable()) displayThread.join();
    timeEndPeriod(1);
    ImGuiUI::Get().Stop();
    return 0;
}
