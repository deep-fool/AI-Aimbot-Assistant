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
static int  open_test = 1;
static ScreenCapture              g_capture;
static std::atomic<bool>          g_running{ false }; 
static std::atomic<bool>          g_requestQuit{ false };

static std::atomic<bool>          g_capParked{ false };
static std::atomic<bool>          g_dispParked{ false };

static std::atomic<bool>          g_requestResume{ false };
static std::atomic<bool>          g_requestPause{ false };
std::atomic<bool>                 g_requestRegionApply{ false };

static std::shared_ptr<ImageData> g_latestFrame{ nullptr };
static std::mutex                 g_frameMutex;
static std::atomic<double>        g_captureFPS{ 0.0 };

static int                        g_captureRadius = 112;

std::atomic<bool>                 g_paused{ false };
static const int MODEL_INPUT_SIZE = 224;
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
static int   y_offset = 12;
static int   MAX_LOST_FRAMES = 6;
static float PREDICT_TIME = 0.007f;
static float VELOCITY_ALPHA = 0.35f;
static float MAX_VELOCITY = 800.0f;
static std::vector<int> g_categories = { 0, 5 };
static float conf = 0.25f;
static float SAME_TARGET_DIST_SQ = 2500.0f;
static bool  g_autoFireEnabled = true;
static bool  fireNow = false;
static float g_fireRadius = 3.0f;
static float CFG_AIM_OFFSET_PCT = -30.0f;

// ── 自瞄触发键 ───────────────────────────────────────────────
static bool g_aimOnLeft = false;   // 左键触发自瞄
static bool g_aimOnRight = true;    // 右键触发自瞄

// ── 急停射击（改为"屏蔽按键"实现）─────────────────────────────
static bool g_counterStrafing = false;

// ── 键盘屏蔽方式 ─────────────────────────────────────────────
//   0=关闭  1=KMBox屏蔽(kmNet_mask_keyboard)  2=系统Hook屏蔽(WH_KEYBOARD_LL)
enum KbdMaskMode { MASK_OFF = 0, MASK_KMBOX = 1, MASK_HOOK = 2 };
static int g_kbdMaskMode = MASK_HOOK;   // 启动时按 KMBox 是否连接覆盖（见 main）

// 当前"应被系统 Hook 吞掉"的按键集合（每个 VK 一个原子标志）。
// CaptureLoop/DisplayLoop 设置它，低级钩子线程读取它。
static std::atomic<bool> g_maskA{ false }, g_maskD{ false },
g_maskW{ false }, g_maskS{ false };

// ── KMBox 连接状态 ───────────────────────────────────────────
bool g_kmConnected = false;

// ── 键盘发送：KMBox优先，失败降级SendInput ───────────────────
static inline void KeyDown(WORD vk) {
    if (g_kmConnected) { kmNet_enc_keydown((int)vk); return; }
    INPUT in = {}; in.type = INPUT_KEYBOARD;
    in.ki.wVk = vk; in.ki.dwFlags = 0;
    SendInput(1, &in, sizeof(INPUT));
}
static inline void KeyUp(WORD vk) {
    if (g_kmConnected) { kmNet_enc_keyup((int)vk); return; }
    INPUT in = {}; in.type = INPUT_KEYBOARD;
    in.ki.wVk = vk; in.ki.dwFlags = KEYEVENTF_KEYUP;
    SendInput(1, &in, sizeof(INPUT));
}

// ═══════════════════════════════════════════════════════════════════
//  键盘屏蔽（急停用）
//  ------------------------------------------------------------------
//  两种实现，由 g_kbdMaskMode 选择：
//   - MASK_KMBOX: 调 kmNet_mask_keyboard / kmNet_unmask_keyboard（HID 扫描码）
//   - MASK_HOOK : 设置 g_maskX 原子，由 WH_KEYBOARD_LL 钩子吞掉对应键
//  统一入口 SetKeyMasked(vk,on)；上层逻辑不关心底层用哪种。
// ═══════════════════════════════════════════════════════════════════

// vk -> 系统 Hook 原子标志指针（仅 WASD）
static std::atomic<bool>* HookFlagFor(WORD vk) {
    switch (vk) {
    case 'A': return &g_maskA;
    case 'D': return &g_maskD;
    case 'W': return &g_maskW;
    case 'S': return &g_maskS;
    default:  return nullptr;
    }
}

// Windows VK -> USB HID 键盘 Usage ID（KMBox 固件用 HID 扫描码，不是 Windows VK）
//   A=0x04 B=0x05 ... 这里只需 WASD。返回 0 表示无映射。
static short VkToHid(WORD vk) {
    switch (vk) {
    case 'A': return 0x04;
    case 'D': return 0x07;
    case 'W': return 0x1A;
    case 'S': return 0x16;
    default:  return 0;
    }
}

// 屏蔽/解除屏蔽一个按键。on=true 屏蔽，on=false 解除。
static void SetKeyMasked(WORD vk, bool on) {
    switch (g_kbdMaskMode) {
    case MASK_KMBOX:
        if (g_kmConnected) {
            short hid = VkToHid(vk);
            if (hid) {
                if (on) kmNet_mask_keyboard(hid);
                else    kmNet_unmask_keyboard(hid);
            }
        }
        break;
    case MASK_HOOK:
        if (auto* f = HookFlagFor(vk)) f->store(on, std::memory_order_release);
        break;
    case MASK_OFF:
    default:
        break;
    }
}

// 把所有 WASD 的屏蔽一次性解除（停止/暂停/丢失目标/松开触发键时调用）。
// 注意：KMBox 端无条件 unmask（即使当前模式不是 KMBox，也防止切模式后残留屏蔽）。
static void ClearAllKeyMasks() {
    // 系统 Hook 标志
    g_maskA.store(false, std::memory_order_release);
    g_maskD.store(false, std::memory_order_release);
    g_maskW.store(false, std::memory_order_release);
    g_maskS.store(false, std::memory_order_release);
    // KMBox 端解除（HID 扫描码；防止从 KMBox 模式切走后仍被硬件屏蔽）
    if (g_kmConnected) {
        kmNet_unmask_keyboard(VkToHid('A'));
        kmNet_unmask_keyboard(VkToHid('D'));
        kmNet_unmask_keyboard(VkToHid('W'));
        kmNet_unmask_keyboard(VkToHid('S'));
    }
}

// 供 UI 调用（非 static）：切换屏蔽方式时清掉所有残留屏蔽
void UIClearKeyMasks() { ClearAllKeyMasks(); }

// ── 低级键盘钩子（WH_KEYBOARD_LL）：按 g_maskX 吞键 ───────────────
//   只有在 MASK_HOOK 模式且对应 g_maskX=true 时，才拦截(返回1)该按键，
//   游戏/系统都收不到。其余一律放行。钩子运行在自己的消息线程。
static HHOOK  g_llKbdHook = nullptr;
static std::thread g_hookThread;
static std::atomic<DWORD> g_hookThreadId{ 0 };

static LRESULT CALLBACK LowLevelKbdProc(int code, WPARAM w, LPARAM l) {
    if (code == HC_ACTION && g_kbdMaskMode == MASK_HOOK) {
        auto* kb = (KBDLLHOOKSTRUCT*)l;
        std::atomic<bool>* f = HookFlagFor((WORD)kb->vkCode);
        if (f && f->load(std::memory_order_acquire)) {
            // 不区分按下/抬起，全部吞掉：屏蔽期间该键对系统完全不可见
            return 1;
        }
    }
    return CallNextHookEx(nullptr, code, w, l);
}

static void StartKbdHook() {
    g_hookThread = std::thread([]() {
        g_hookThreadId.store(GetCurrentThreadId(), std::memory_order_release);
        g_llKbdHook = SetWindowsHookExW(
            WH_KEYBOARD_LL, LowLevelKbdProc, GetModuleHandleW(nullptr), 0);
        if (!g_llKbdHook) {
            PushLog("[屏蔽] 低级键盘钩子安装失败");
        }
        else {
            PushLog("[屏蔽] 系统Hook已就绪");
        }
        MSG msg;
        // 钩子必须有消息循环来分发 LL 回调
        while (GetMessageW(&msg, nullptr, 0, 0) > 0) {
            TranslateMessage(&msg); DispatchMessageW(&msg);
        }
        if (g_llKbdHook) { UnhookWindowsHookEx(g_llKbdHook); g_llKbdHook = nullptr; }
        });
}

static void StopKbdHook() {
    DWORD tid = g_hookThreadId.load(std::memory_order_acquire);
    if (tid) PostThreadMessageW(tid, WM_QUIT, 0, 0);
    if (g_hookThread.joinable()) g_hookThread.join();
}

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
static int model_way = 0;
static int inference_engine_mode = 0;
static bool open_findcolor = false;
int  g_modelInputSize = 224;
char g_modelPath[512];
char g_kmIP[64] = "192.168.2.188";
char g_kmPort[16] = "1000";
char g_kmMAC[32] = "25ABDBB2";

// ── 找色参数 ─────────────────────────────────────────────────
float fc_h1lo = 0, fc_h1hi = 3, fc_s1lo = 100, fc_s1hi = 255, fc_v1lo = 120, fc_v1hi = 255;
float fc_h2lo = 177, fc_h2hi = 180, fc_s2lo = 100, fc_s2hi = 255, fc_v2lo = 100, fc_v2hi = 255;
int   fc_roi_x = -80, fc_roi_y = -80, fc_roi_w = 160, fc_roi_h = 160;
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
static void CaptureInitToCurrentSize()
{
    int w = g_capW.load(std::memory_order_acquire);
    int h = g_capH.load(std::memory_order_acquire);
    int sw = GetSystemMetrics(SM_CXSCREEN);
    int sh = GetSystemMetrics(SM_CYSCREEN);
    int tx = (sw - w) / 2, ty = (sh - h) / 2;

    g_capture.Reset();
    g_backend = 1;
    if (!g_capture.Init(w, h, g_backend)) {
        g_backend = 0;
        if (!g_capture.Init(w, h, g_backend)) {
            PushLog("[截图] Init 失败(DXGI+GDI均失败)");
            return;
        }
    }
    g_capture.SetRegion(tx, ty, w, h);

    // 丢弃任何旧尺寸残帧，DisplayLoop 醒来后只会拿到新尺寸帧
    { std::lock_guard<std::mutex> lk(g_frameMutex); g_latestFrame.reset(); }

    char buf[128];
    snprintf(buf, sizeof(buf), "[截图] 初始化 %dx%d 区域(%d,%d) 后端:%s",
        w, h, tx, ty, g_backendNames[g_backend]);
    PushLog(buf);
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
            CaptureInitToCurrentSize();
            inited = true;
            frameCount = 0;
            lastTime = std::chrono::high_resolution_clock::now();
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
    int screenW = GetSystemMetrics(SM_CXSCREEN);
    int screenH = GetSystemMetrics(SM_CYSCREEN);

    bool      hasLockedTarget = false;
    int       lostFrameCount = 0;
    int       lockedCX = 0, lockedCY = 0;
    Detection lockedTarget;
    float     vx = 0.f, vy = 0.f;
    bool      velocityReady = false;

    // ── 急停屏蔽状态（记录我们当前屏蔽了哪些键）────────────────
    bool cs_maskD = false, cs_maskA = false;
    bool cs_maskS = false, cs_maskW = false;

    // 解除所有急停屏蔽的 lambda（停止/暂停/松键/丢失目标时调用）
    auto csReleaseAll = [&]() {
        if (cs_maskD) { SetKeyMasked('D', false); cs_maskD = false; }
        if (cs_maskA) { SetKeyMasked('A', false); cs_maskA = false; }
        if (cs_maskS) { SetKeyMasked('S', false); cs_maskS = false; }
        if (cs_maskW) { SetKeyMasked('W', false); cs_maskW = false; }
        // 兜底：无条件清掉底层所有屏蔽（防止切模式/异常残留）
        ClearAllKeyMasks();
        };

    auto lastFrameTime = std::chrono::high_resolution_clock::now();
    auto lastFpsTime = std::chrono::high_resolution_clock::now();

    // 推理帧计数
    int   inferCount = 0;
    auto  inferFpsTimer = std::chrono::high_resolution_clock::now();

    bool wasRunning = false;  // 上一帧是否在运行（用于检测恢复时机）

    while (!g_requestQuit.load())
    {
        // ── 未运行：停在安全点，宣告 parked，绝不用旧帧/不调用 Inference ──
        if (!g_running.load(std::memory_order_acquire)) {
            if (wasRunning) {
                // 刚从运行变为停止：释放急停键，彻底重置跟踪状态
                csReleaseAll();
                hasLockedTarget = false; lostFrameCount = 0;
                vx = vy = 0.f; velocityReady = false;
                cs_maskD = cs_maskA = cs_maskS = cs_maskW = false;
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
            if (open_findcolor) {
                cv::Rect roi(screenCenterX + fc_roi_x, screenCenterY + fc_roi_y,
                    fc_roi_w, fc_roi_h);
                roi &= cv::Rect(0, 0, frameMat.cols, frameMat.rows);
                cv::Mat roiMat = frameMat(roi);
                FindColor(roiMat);
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

                    // ── 急停射击（屏蔽按键实现）─────────────────
                    // 条件：按下自瞄触发键 AND 有锁定目标（已在此分支内）。
                    // 逻辑：把玩家当前按住的 WASD 直接屏蔽，让游戏收不到 →
                    //       角色立即停止该方向移动（急停）。
                    if (g_counterStrafing && g_kbdMaskMode != MASK_OFF) {
                        // ★ 反馈震荡防护（关键）：
                        //   系统Hook模式下，一旦屏蔽某键，LL钩子会吞掉它，
                        //   连我们自己的 Raw Input / GetAsyncKeyState 也读不到了，
                        //   于是会误判"玩家松手"→解除→又检测到→再屏蔽，产生震荡。
                        //   解决：已屏蔽的键采用"latch"——本次自瞄按住期间保持屏蔽，
                        //   不再因读不到按键而解除；只有在自瞄键松开（else 分支）
                        //   或丢失目标/暂停（csReleaseAll）时才统一解除。
                        bool pA = (g_rawA.load() || (GetAsyncKeyState('A') & 0x8000) != 0);
                        bool pD = (g_rawD.load() || (GetAsyncKeyState('D') & 0x8000) != 0);
                        bool pW = (g_rawW.load() || (GetAsyncKeyState('W') & 0x8000) != 0);
                        bool pS = (g_rawS.load() || (GetAsyncKeyState('S') & 0x8000) != 0);

                        // 按下且尚未屏蔽 → 开始屏蔽并 latch
                        if (pA && !cs_maskA) { SetKeyMasked('A', true); cs_maskA = true; PushLog("[急停] 屏蔽 A"); }
                        if (pD && !cs_maskD) { SetKeyMasked('D', true); cs_maskD = true; PushLog("[急停] 屏蔽 D"); }
                        if (pW && !cs_maskW) { SetKeyMasked('W', true); cs_maskW = true; PushLog("[急停] 屏蔽 W"); }
                        if (pS && !cs_maskS) { SetKeyMasked('S', true); cs_maskS = true; PushLog("[急停] 屏蔽 S"); }
                        // 已屏蔽的键保持 latch，不在此处解除（见上注释）
                    }
                }
                else {
                    // 自瞄键松开：立即释放急停键
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
            float fps = 1000.f / std::chrono::duration<float, std::milli>(
                nowFps - lastFpsTime).count();
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

    // 不再设定默认模型路径——用户必须在 UI 中选择模型后才能开始运行
    g_modelPath[0] = '\0';

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

    // ── 键盘屏蔽默认值：有 KMBox 连接默认用 KMBox 屏蔽，否则用系统 Hook。
    //    仅在启动时设定一次；用户之后可在 UI 自由切换，不会被覆盖。
    g_kbdMaskMode = g_kmConnected ? MASK_KMBOX : MASK_HOOK;
    PushLog(g_kmConnected ? "[屏蔽] 默认: KMBox屏蔽" : "[屏蔽] 默认: 系统Hook屏蔽");

    SetMoveMethod(g_kmConnected ? 0 : 1);

    g_capW.store(g_captureRadius * 2, std::memory_order_release);
    g_capH.store(g_captureRadius * 2, std::memory_order_release);

    {
        char buf[128];
        snprintf(buf, sizeof(buf), "[系统] 截图%dx%d 模型输入%dx%d",
            g_capW.load(), g_capH.load(), g_modelInputSize, g_modelInputSize);
        PushLog(buf);
    }

    // 不在启动时自动加载模型——用户必须先在 UI 中选择模型路径，
    // 然后点击"开始"才会加载并运行。避免模型路径无效时直接报错退出。
    PushLog("[系统] 启动完成，请选择模型文件后点击[开始]");

    timeBeginPeriod(1);
    StartRawInput();
    StartKbdHook();          // 低级键盘钩子（系统Hook屏蔽用，常驻）
    ImGuiUI::Get().Start();

    // CaptureLoop 和 DisplayLoop 是常驻线程；g_running=false 时它们 park 在安全点。
    // 启动时 g_running=false，两线程先停在 parked。
    std::thread captureThread(CaptureLoop);
    std::thread displayThread([] { DisplayLoop(); });

    // 设定初始尺寸（此时两线程已 parked，安全）
    g_capW.store(g_captureRadius * 2, std::memory_order_release);
    g_capH.store(g_captureRadius * 2, std::memory_order_release);

    // ── 默认不自动运行，保持在暂停状态。用户选择模型后点击"开始"才会放行。
    //    此时 g_running=false，两线程在 parked 安全点等待。
    g_paused.store(true, std::memory_order_release);
    PushLog("[系统] 就绪——请选择模型文件后点击[开始]");

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

        // ── (C) 开始/继续请求：全停 → 校验模型 → 重载引擎/KMBox → 放行 ──
        if (g_requestResume.exchange(false)) {
            PushLog("[系统] 收到开始/继续，校验模型…");

            // ★ 校验模型路径：未选择模型文件则拒绝运行，防止引擎加载失败导致崩溃
            if (g_modelPath[0] == '\0') {
                PushLog("[错误] 未选择模型文件！请先在[设备]页浏览选择模型，再点击开始");
                continue;   // 保持暂停，不放行
            }
            if (GetFileAttributesA(g_modelPath) == INVALID_FILE_ATTRIBUTES) {
                char buf[512];
                snprintf(buf, sizeof(buf), "[错误] 模型文件不存在: %s", g_modelPath);
                PushLog(buf);
                continue;   // 保持暂停，不放行
            }

            StopAndWaitParked();                  // ★ 确定性全停
            if (g_requestQuit.load()) break;

            // 重载推理引擎（两线程已 parked，g_detector 无人使用）
            try {
                g_detector.reset();
                g_detector = std::make_unique<Detector>(
                    std::string(g_modelPath), model_way,
                    static_cast<InferenceEngine>(inference_engine_mode));
                g_detector->SetInputSize(g_modelInputSize);
                char buf[128];
                snprintf(buf, sizeof(buf), "[引擎] 加载成功 %dx%d", g_modelInputSize, g_modelInputSize);
                PushLog(buf);
            }
            catch (const std::exception& e) {
                char buf[256];
                snprintf(buf, sizeof(buf), "[引擎] 加载失败: %s", e.what());
                PushLog(buf);
                // 加载失败：保持暂停，不放行，避免无引擎空跑
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
    ClearAllKeyMasks();      // 确保退出时不残留任何键盘屏蔽
    StopRawInput();
    StopKbdHook();           // 卸载低级键盘钩子
    if (captureThread.joinable()) captureThread.join();
    if (displayThread.joinable()) displayThread.join();
    timeEndPeriod(1);
    ImGuiUI::Get().Stop();
    return 0;
}
