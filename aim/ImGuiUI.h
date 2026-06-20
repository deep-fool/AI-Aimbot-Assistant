#pragma once
// ============================================================
//  ImGuiUI.h  v3  —  ImGui DX11 多标签控制面板
//
//  标签页：
//    [监控]   FPS / 锁定 / 置信度 / 距离 + 折线图
//    [PID]    Kp/Ki/Kd / 预测权重 / 输出参数
//    [自瞄]   瞄点 / 死区 / 开火 / 跟踪
//    [高级]   限速表 / 贝塞尔 / 震荡抑制
//    [找色]   HSV范围 / 截图范围 / 瞄准范围 / 开关
//    [设备]   KMBox / 引擎 / 模型 / 类别ID / 功能开关
//    [配置]   保存/加载配置
// ============================================================

#include "imgui/imgui.h"
#include "imgui/imgui_impl_win32.h"
#include "imgui/imgui_impl_dx11.h"
#include <d3d11.h>
#include <dwmapi.h>
#include <windows.h>
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dwmapi.lib")

#include <atomic>
#include <thread>
#include <deque>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

#include "pid.h"
#include <commdlg.h>
#pragma comment(lib,"comdlg32.lib")

// ── main.cpp 可调参数 extern ──────────────────────────────────
extern int              y_offset;
extern int              MAX_LOST_FRAMES;
extern float            PREDICT_TIME;
extern float            VELOCITY_ALPHA;
extern float            MAX_VELOCITY;
extern float            conf;
extern float            SAME_TARGET_DIST_SQ;
extern bool             g_autoFireEnabled;
extern float            g_fireRadius;
extern float            CFG_AIM_OFFSET_PCT;
extern int              g_captureRadius;
extern std::atomic<bool> g_requestRegionApply;   // 改截图半径松手后触发
extern int              open_test;
extern int              model_way;
extern int              inference_engine_mode;
extern bool             open_findcolor;
extern bool             g_aimOnLeft;
extern bool             g_aimOnRight;
extern bool             g_counterStrafing;
extern int              g_csBackend;     // 0=关 1=KMBox反向 2=系统反向
// UI 切换急停后端时调用（占位，定义在 main.cpp）
extern void UIClearKeyMasks();
extern bool             g_kmConnected;
extern std::atomic<float> g_captureMs;
extern std::atomic<float> g_inferMs;
extern std::atomic<float> g_inferFPS;
#include <mutex>
#include <deque>
extern std::mutex              g_logMutex;
extern std::deque<std::string> g_logBuf;
extern std::atomic<bool> g_running;
extern std::atomic<bool> g_paused;
extern std::atomic<bool> g_requestResume;
extern std::atomic<bool> g_requestPause;
extern std::atomic<bool> g_requestQuit;

// ── 找色参数（定义在 main.cpp，这里 extern）─────────────────
extern float fc_h1lo, fc_h1hi, fc_s1lo, fc_s1hi, fc_v1lo, fc_v1hi;
extern float fc_h2lo, fc_h2hi, fc_s2lo, fc_s2hi, fc_v2lo, fc_v2hi;
extern int   fc_roi_x, fc_roi_y, fc_roi_w, fc_roi_h;
extern float fc_aim_scale;
extern int   fc_min_area;

// ── 类别优先级列表（定义在 main.cpp）───────────────────────
extern std::vector<int> g_categories;

// ── 模型输入尺寸（定义在 main.cpp）─────────────────────────
extern int g_modelInputSize;

// ── 配置保存路径（定义在 main.cpp）─────────────────────────
extern char g_modelPath[512];
extern char g_kmIP[64];
extern char g_kmPort[16];
extern char g_kmMAC[32];

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(
    HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);

// ============================================================
class ImGuiUI {
public:
    static ImGuiUI& Get() { static ImGuiUI inst; return inst; }

    void PushTelemetry(float fps, bool locked, float conf_v,
        float dist, float vx, float vy, int classId)
    {
        m_fps = fps; m_locked = locked; m_conf = conf_v;
        m_dist = dist; m_vx = vx; m_vy = vy; m_classId = classId;
    }

    void Start() {
        if (m_thread.joinable()) return;
        m_quit = false;
        m_thread = std::thread([this] { Run(); });
    }

    void Stop() {
        m_quit = true;
        if (m_thread.joinable()) m_thread.join();
    }

private:
    ImGuiUI() = default;

    std::atomic<float> m_fps{ 0 }, m_conf{ 0 }, m_dist{ 0 }, m_vx{ 0 }, m_vy{ 0 };
    std::atomic<bool>  m_locked{ false };
    std::atomic<int>   m_classId{ -1 };
    std::atomic<bool>  m_quit{ false };
    std::thread        m_thread;

    ID3D11Device* m_dev = nullptr;
    ID3D11DeviceContext* m_ctx = nullptr;
    IDXGISwapChain* m_sc = nullptr;
    ID3D11RenderTargetView* m_rtv = nullptr;
    HWND                    m_hwnd = nullptr;

    static constexpr int HIST = 120;
    std::deque<float> m_fpsQ, m_vxQ, m_vyQ;

    // 类别 ID 编辑缓冲
    char m_catBuf[256] = "0,5";
    char m_saveMsg[128] = "";
    float m_saveMsgTimer = 0.f;

    // ── UI 线程 ──────────────────────────────────────────────
    void Run()
    {
        WNDCLASSEXW wc{ sizeof(wc) };
        wc.style = CS_CLASSDC; wc.lpfnWndProc = WndProc;
        wc.hInstance = GetModuleHandleW(nullptr);
        wc.lpszClassName = L"AimUI_v3";
        wc.hCursor = LoadCursor(nullptr, IDC_ARROW);
        RegisterClassExW(&wc);

        m_hwnd = CreateWindowExW(0, L"AimUI_v3", L"Aim Control Panel",
            WS_OVERLAPPEDWINDOW, 120, 80, 1060, 860,
            nullptr, nullptr, wc.hInstance, this);
        if (!m_hwnd || !InitDX11()) {
            UnregisterClassW(wc.lpszClassName, wc.hInstance); return;
        }
        MARGINS mg{ -1,-1,-1,-1 };
        DwmExtendFrameIntoClientArea(m_hwnd, &mg);
        ShowWindow(m_hwnd, SW_SHOW); UpdateWindow(m_hwnd);

        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        io.IniFilename = nullptr;

        static const ImWchar ranges[] = { 0x0020,0x00FF,0x4E00,0x9FAF,0 };
        ImFontConfig fc2; fc2.OversampleH = 2; fc2.OversampleV = 2;
        if (GetFileAttributesA("C:\\Windows\\Fonts\\msyh.ttc") != INVALID_FILE_ATTRIBUTES)
            io.Fonts->AddFontFromFileTTF("C:\\Windows\\Fonts\\msyh.ttc", 18.f, &fc2, ranges);
        else
            io.Fonts->AddFontDefault();

        Theme();
        ImGui_ImplWin32_Init(m_hwnd);
        ImGui_ImplDX11_Init(m_dev, m_ctx);

        // 初始化类别缓冲
        SyncCatBufFromVec();

        const float BG[4] = { 0.08f,0.09f,0.10f,1.f };
        while (!m_quit) {
            MSG msg;
            while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
                TranslateMessage(&msg); DispatchMessageW(&msg);
                if (msg.message == WM_QUIT) m_quit = true;
            }
            if (m_quit) break;
            ImGui_ImplDX11_NewFrame();
            ImGui_ImplWin32_NewFrame();
            ImGui::NewFrame();
            Draw();
            ImGui::Render();
            m_ctx->OMSetRenderTargets(1, &m_rtv, nullptr);
            m_ctx->ClearRenderTargetView(m_rtv, BG);
            ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());
            m_sc->Present(1, 0);
        }
        ImGui_ImplDX11_Shutdown();
        ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
        FreeDX11();
        DestroyWindow(m_hwnd);
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
    }

    // ── 主绘制 ───────────────────────────────────────────────
    void Draw()
    {
        ImGuiIO& io = ImGui::GetIO();
        ImGui::SetNextWindowPos({ 0,0 });
        ImGui::SetNextWindowSize(io.DisplaySize);
        ImGui::SetNextWindowBgAlpha(0.f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, { 0,0 });
        ImGui::Begin("##root", nullptr,
            ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
            ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoBringToFrontOnFocus |
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
        ImGui::PopStyleVar();

        PushHist(m_fpsQ, m_fps.load());
        PushHist(m_vxQ, m_vx.load());
        PushHist(m_vyQ, m_vy.load());

        // 倒计时消息
        if (m_saveMsgTimer > 0.f) {
            m_saveMsgTimer -= io.DeltaTime;
            if (m_saveMsgTimer < 0.f) m_saveMsgTimer = 0.f;
        }

        TopBar(io);

        // 标签栏样式
        ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, { 16,9 });
        ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 2,0 });
        ImGui::PushStyleColor(ImGuiCol_Tab, C(.10f, .11f, .13f, 1));
        ImGui::PushStyleColor(ImGuiCol_TabHovered, C(.16f, .18f, .21f, 1));
        ImGui::PushStyleColor(ImGuiCol_TabActive, C(.00f, .55f, .28f, 1));
        ImGui::PushStyleColor(ImGuiCol_TabUnfocusedActive, C(.00f, .42f, .21f, 1));

        if (ImGui::BeginTabBar("##tabs")) {
            Tab("  监控  ", [this] {PageMonitor(); });
            Tab("  PID  ", [this] {PagePID(); });
            Tab("  自瞄  ", [this] {PageAim(); });
            Tab("  高级  ", [this] {PageAdv(); });
            Tab("  找色  ", [this] {PageFindColor(); });
            Tab("  设备  ", [this] {PageDevice(); });
            Tab("  配置  ", [this] {PageConfig(); });
            ImGui::EndTabBar();
        }
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor(4);
        ImGui::End();
    }

    template<typename Fn>
    void Tab(const char* label, Fn fn) {
        if (ImGui::BeginTabItem(label)) {
            ImGui::PopStyleVar(2); ImGui::PopStyleColor(4);
            fn();
            ImGui::EndTabItem();
            ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, { 16,9 });
            ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, { 2,0 });
            ImGui::PushStyleColor(ImGuiCol_Tab, C(.10f, .11f, .13f, 1));
            ImGui::PushStyleColor(ImGuiCol_TabHovered, C(.16f, .18f, .21f, 1));
            ImGui::PushStyleColor(ImGuiCol_TabActive, C(.00f, .55f, .28f, 1));
            ImGui::PushStyleColor(ImGuiCol_TabUnfocusedActive, C(.00f, .42f, .21f, 1));
        }
    }

    // ── 顶栏 ─────────────────────────────────────────────────
    void TopBar(const ImGuiIO& io)
    {
        bool locked = m_locked; float fps = m_fps;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##top", { 0,48 }, false,
            ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);

        ImGui::SetCursorPos({ 14,14 });
        ImGui::TextColored(locked ? C(0, .9f, .46f, 1) : C(.37f, .38f, .40f, 1), "*");
        ImGui::SameLine(0, 8); ImGui::SetCursorPosY(13);
        ImGui::Text("Aim Control Panel");
        ImGui::SameLine(0, 20); ImGui::SetCursorPosY(13);
        ImGui::TextColored(locked ? C(0, .9f, .46f, 1) : C(.5f, .5f, .5f, 1),
            locked ? "已锁定" : "未检测");

        // 顶栏右侧按钮区：暂停 | 继续 | 退出
        bool running = g_running.load();
        bool paused = g_paused.load();
        float bx = io.DisplaySize.x - 390;
        ImGui::SameLine(bx); ImGui::SetCursorPosY(11);

        if (!running && !paused) {
            // 未运行：绿色"开始"
            ImGui::PushStyleColor(ImGuiCol_Button, C(.00f, .45f, .23f, 1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, C(.00f, .62f, .31f, 1));
            if (ImGui::Button(">  开始##start", { 100,26 })) g_requestResume = true;
            ImGui::PopStyleColor(2);
        }
        else if (running && !paused) {
            // 运行中：黄色"暂停"
            ImGui::PushStyleColor(ImGuiCol_Button, C(.55f, .40f, .00f, 1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, C(.75f, .55f, .00f, 1));
            if (ImGui::Button("||  暂停##pause", { 100,26 })) {
                // 通过主控线程做"确定性全停"，不在 UI 线程直接动 g_running
                g_requestPause = true;
            }
            ImGui::PopStyleColor(2);
        }
        else {
            // 暂停中：绿色"继续"
            ImGui::PushStyleColor(ImGuiCol_Button, C(.00f, .45f, .23f, 1));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, C(.00f, .62f, .31f, 1));
            if (ImGui::Button(">  继续##resume", { 100,26 })) g_requestResume = true;
            ImGui::PopStyleColor(2);
        }
        ImGui::SameLine(0, 8); ImGui::SetCursorPosY(11);
        ImGui::PushStyleColor(ImGuiCol_Button, C(.50f, .08f, .05f, 1));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, C(.75f, .12f, .08f, 1));
        if (ImGui::Button("X  退出##quit", { 90,26 })) {
            g_requestQuit = true; g_paused = false; g_running = false;
        }
        ImGui::PopStyleColor(2);
        ImGui::SameLine(0, 10); ImGui::SetCursorPosY(14);
        if (!running && !paused) ImGui::TextColored(C(.5f, .5f, .5f, 1), "未运行");
        else if (paused)           ImGui::TextColored(C(.9f, .7f, .0f, 1), "已暂停");
        else                       ImGui::TextDisabled("%.0f fps", fps);

        ImGui::EndChild();
        ImGui::PopStyleColor();
        // 分割线
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        dl->AddLine({ p.x,p.y }, { p.x + io.DisplaySize.x,p.y }, IM_COL32(42, 45, 48, 255));
    }

    // ============================================================
    //  页：监控
    // ============================================================
    void PageMonitor()
    {
        Pad();
        float fps = m_fps; bool locked = m_locked;
        float cv2 = m_conf; float dist = m_dist;
        float vx = m_vx, vy = m_vy; int cls = m_classId;
        float W = ImGui::GetContentRegionAvail().x;
        float cw = (W - 30) / 4.f;

        // 4 卡片
        auto Card = [&](const char* id, const char* lbl,
            const char* big, ImVec4 bigC,
            const char* s1, const char* s2 = "") {
                ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
                ImGui::BeginChild(id, { cw,96 }, true);
                Small(lbl);
                ImGui::TextColored(bigC, "%s", big);
                Small(s1); if (s2[0]) Small(s2);
                ImGui::EndChild(); ImGui::PopStyleColor();
            };

        char fpsBuf[32], coBuf[32], diBuf[32], vBuf[48], clBuf[32];
        snprintf(fpsBuf, 32, "%.0f fps", fps);
        snprintf(coBuf, 32, locked ? "%.2f" : "--", cv2);
        snprintf(diBuf, 32, locked ? "%.0f px" : "--", dist);
        snprintf(vBuf, 48, locked ? "Vx %.0f  Vy %.0f" : "Vx --  Vy --", vx, vy);
        snprintf(clBuf, 32, locked ? "Class %d" : "--", cls);
        const char* fh = fps > 100 ? "流畅" : fps > 60 ? "良好" : "低帧率";

        ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 8);
        Card("##c1", "推理 FPS", fpsBuf, C(0, .9f, .46f, 1), fh);
        ImGui::SameLine(0, 8);
        Card("##c2", "锁定目标", locked ? "已锁定" : "未检测",
            locked ? C(0, .9f, .46f, 1) : C(.5f, .5f, .5f, 1), clBuf);
        ImGui::SameLine(0, 8);

        // 置信度带进度条
        {
            ImVec4 cc = cv2 > .7f ? C(0, .9f, .46f, 1) : cv2 > .45f ? C(1, .7f, 0, 1) : C(.9f, .3f, .3f, 1);
            ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
            ImGui::BeginChild("##c3", { cw,96 }, true);
            Small("置信度");
            ImGui::TextColored(cc, "%s", coBuf);
            ImGui::PushStyleColor(ImGuiCol_PlotHistogram, cc);
            ImGui::ProgressBar(locked ? cv2 : 0.f, { -1,5 }, "");
            ImGui::PopStyleColor();
            ImGui::EndChild(); ImGui::PopStyleColor();
        }
        ImGui::SameLine(0, 8);
        Card("##c4", "距中心", diBuf, C(.91f, .92f, .93f, 1), vBuf);

        ImGui::Spacing();

        // ── 耗时状态栏 ──────────────────────────────────────
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.08f, .09f, .10f, 1));
        ImGui::BeginChild("##perf", { 0,36 }, false, ImGuiWindowFlags_NoScrollbar);
        ImGui::SetCursorPos({ 10,9 });
        ImGui::TextColored(C(0, .9f, .46f, 1),
            "推理 %.1f ms  |  截图 %.1f ms  |  推理帧率 %.0f fps",
            g_inferMs.load(), g_captureMs.load(), g_inferFPS.load());
        ImGui::SameLine(0, 24);
        ImGui::TextColored(
            g_kmConnected ? C(0, .9f, .46f, 1) : C(.9f, .5f, .2f, 1),
            "KMBox: %s", g_kmConnected ? "已连接" : "未连接(SendInput)");
        ImGui::EndChild();
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // ── 折线图（高度自适应剩余空间的一半）───────────────
        float avail = ImGui::GetContentRegionAvail().y;
        float chH = avail * 0.42f;   // 图表占可用高度42%
        float chW = (W - 10) / 2.f;

        ChartBox("##ch1", "FPS 历史", m_fpsQ, 0, 250, chW, chH, C(0, .9f, .46f, 1));
        ImGui::SameLine(0, 8);
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##ch2", { chW,chH }, true);
        Small("速度分量 Vx / Vy");
        if (!m_vxQ.empty()) {
            float pH = ImGui::GetContentRegionAvail().y * 0.5f - 2;
            std::vector<float> vxv(m_vxQ.begin(), m_vxQ.end());
            std::vector<float> vyv(m_vyQ.begin(), m_vyQ.end());
            ImGui::PushStyleColor(ImGuiCol_PlotLines, C(.29f, .61f, 1, 1));
            ImGui::PlotLines("##vx", vxv.data(), (int)vxv.size(), 0, "Vx", -500, 500, { -1,pH });
            ImGui::PopStyleColor();
            ImGui::PushStyleColor(ImGuiCol_PlotLines, C(1, .42f, .42f, 1));
            ImGui::PlotLines("##vy", vyv.data(), (int)vyv.size(), 0, "Vy", -500, 500, { -1,pH });
            ImGui::PopStyleColor();
        }
        ImGui::EndChild(); ImGui::PopStyleColor();

        ImGui::Spacing();

        // ── 控制台日志（填满剩余空间）───────────────────────
        float logH = ImGui::GetContentRegionAvail().y - 4;
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.05f, .06f, .07f, 1));
        ImGui::PushStyleColor(ImGuiCol_Text, C(.60f, .90f, .60f, 1));
        ImGui::BeginChild("##log", { 0,logH }, true, ImGuiWindowFlags_HorizontalScrollbar);
        {
            std::lock_guard<std::mutex> lk(g_logMutex);
            for (const auto& line : g_logBuf)
                ImGui::TextUnformatted(line.c_str());
        }
        if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 20)
            ImGui::SetScrollHereY(1.f);
        ImGui::EndChild();
        ImGui::PopStyleColor(2);
    }

    // ============================================================
    //  页：PID
    // ============================================================
    void PagePID()
    {
        Pad();
        float W = ImGui::GetContentRegionAvail().x;
        float h = (W - 12) * .5f;

        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##pid_g", { h,-86 }, true);
        CT("PID 增益");
        Row("Kp（比例）", &CFG_KP, .01f, 1.f, "%.3f");
        Row("Ki（积分）", &CFG_KI, .001f, .1f, "%.3f");
        Row("Kd（微分）", &CFG_KD, .001f, .05f, "%.3f");
        Sep(); CT("输出参数");
        Row("输出限幅", &CFG_OUT_MAX, 10, 500, "%.0f px/f");
        Row("爬升时间", &CFG_RAMP_MS, 10, 500, "%.0f ms");
        Row("初始缩放", &CFG_INIT_SCALE, .1f, 2, "%.2f");
        ImGui::EndChild(); ImGui::PopStyleColor();

        ImGui::SameLine(0, 12);

        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##pid_p", { h,-86 }, true);
        CT("泰勒预测权重");
        Row("X 权重", &CFG_PRED_WEIGHT_X, 0, 3, "%.2f");
        Row("Y 权重", &CFG_PRED_WEIGHT_Y, 0, 3, "%.2f");
        Row("起效距离", &CFG_PRED_ZERO_DIST, 0, 200, "%.0fpx");
        Row("全效距离", &CFG_PRED_FULL_DIST, 0, 400, "%.0fpx");
        Row("阶数", &CFG_PRED_ORDER, 1, 5, "%.0f");
        Sep(); CT("低通滤波");
        Row("近距alpha", &CFG_FILTER_ALPHA_NEAR, 0, 1, "%.2f");
        Row("近距阈值", &CFG_FILTER_NEAR_DIST, 10, 300, "%.0fpx");
        ImGui::EndChild(); ImGui::PopStyleColor();

        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##pid_s", { 0,0 }, true);
        CT("当前值");
        ImGui::Text("Kp=%.3f  Ki=%.3f  Kd=%.3f    "
            "PredX=%.2f  PredY=%.2f  起效=%.0fpx  全效=%.0fpx",
            CFG_KP, CFG_KI, CFG_KD,
            CFG_PRED_WEIGHT_X, CFG_PRED_WEIGHT_Y,
            CFG_PRED_ZERO_DIST, CFG_PRED_FULL_DIST);
        ImGui::EndChild(); ImGui::PopStyleColor();
    }

    // ============================================================
    //  页：自瞄
    // ============================================================
    void PageAim()
    {
        Pad();
        float W = ImGui::GetContentRegionAvail().x;
        float h = (W - 12) * .5f;

        // ── 左卡：瞄准 + 触发键 + 急停 ──────────────────────
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##aim_b", { h, 0 }, true);
        CT("瞄准参数");
        RowI("Y 偏移", &y_offset, -50, 50, "%d px");
        Row("瞄点偏移%", &CFG_AIM_OFFSET_PCT, -100, 100, "%.0f%%");
        Row("死区半径", &CFG_DZ_DIST, .5f, 30, "%.1fpx");
        Row("积分清零距", &CFG_INTEGRAL_CLEAR_DIST, 1, 50, "%.0fpx");
        Sep(); CT("自动开火");
        ImGui::Checkbox("启用自动开火##af", &g_autoFireEnabled);
        ImGui::Spacing();
        Row("开火半径", &g_fireRadius, .5f, 30, "%.1fpx");
        Sep(); CT("识别");
        Row("置信度阈值", &conf, .05f, .95f, "%.2f");
        // ★ 仅在用户拖完滑块/输入框松手（结束编辑）时才请求应用，避免拖动过程中
        //   频繁触发 DXGI 重载导致卡死。主控线程会安全地停采→改尺寸→reinit→恢复。
        if (RowI("截图半径", &g_captureRadius, 80, 640, "%d px"))
            g_requestRegionApply.store(true, std::memory_order_release);
        ImGui::PushStyleColor(ImGuiCol_Text, C(.5f, .52f, .55f, 1));
        ImGui::SetWindowFontScale(.85f);
        ImGui::Text("  -> 截图边长 %d x %d px",
            g_captureRadius * 2, g_captureRadius * 2);
        ImGui::SetWindowFontScale(1.f);
        ImGui::PopStyleColor();
        Sep(); CT("自瞄触发键");
        ImGui::TextDisabled("系统检测与KMBox满足其一即触发");
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_FrameBg,
            g_aimOnLeft ? C(0, .35f, .18f, 1) : C(.14f, .15f, .17f, 1));
        ImGui::Checkbox("左键自瞄##aimL", &g_aimOnLeft);
        ImGui::PopStyleColor();
        ImGui::SameLine(0, 16);
        ImGui::PushStyleColor(ImGuiCol_FrameBg,
            g_aimOnRight ? C(0, .35f, .18f, 1) : C(.14f, .15f, .17f, 1));
        ImGui::Checkbox("右键自瞄##aimR", &g_aimOnRight);
        ImGui::PopStyleColor();
        if (!g_aimOnLeft && !g_aimOnRight)
            ImGui::TextColored(C(.9f, .3f, .3f, 1), "  警告：至少选一个！");
        Sep(); CT("急停射击");
        ImGui::Checkbox("启用急停射击##cs", &g_counterStrafing);
        ImGui::Spacing();

        // ── 急停注入后端 ─────────────────────────────────────
        ImGui::PushStyleColor(ImGuiCol_Text, C(.75f, .78f, .82f, 1));
        ImGui::Text("反向后端"); ImGui::PopStyleColor();
        ImGui::SameLine(110); ImGui::SetNextItemWidth(-1);
        const char* csItems[] = { "关闭", "KMBox反向", "系统反向" };
        int prevCs = g_csBackend;
        if (ImGui::Combo("##csbackend", &g_csBackend, csItems, 3)) {
            if (g_csBackend != prevCs) {
                // 切换后端：清掉旧后端可能残留的注入键（由 DisplayLoop 在停/松键时释放，
                // 这里调用占位接口保持一致）。
                UIClearKeyMasks();
            }
        }
        ImGui::Spacing();

        // KMBox 后端但未连接时给出警告
        if (g_csBackend == 1 && !g_kmConnected) {
            ImGui::TextColored(C(.9f, .5f, .2f, 1),
                "  KMBox未连接，反向注入不会生效");
        }

        ImGui::PushStyleColor(ImGuiCol_Text, C(.5f, .52f, .55f, 1));
        ImGui::SetWindowFontScale(.85f);
        ImGui::TextWrapped("条件：按下自瞄触发键 且 有锁定目标时，"
            "玩家按 A→注入按 D，D→A，W→S，S→W 抵消移动实现急停。"
            "松开自瞄键 / 目标丢失 / 暂停时自动释放，不会卡键。");
        ImGui::SetWindowFontScale(1.f);
        ImGui::PopStyleColor();
        ImGui::EndChild(); ImGui::PopStyleColor();

        ImGui::SameLine(0, 12);

        // ── 右卡：跟踪参数 ───────────────────────────────────
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##aim_t", { h, 0 }, true);
        CT("目标跟踪");
        RowI("最大丢帧", &MAX_LOST_FRAMES, 1, 30, "%d 帧");
        Row("预测时间", &PREDICT_TIME, .001f, .05f, "%.3fs");
        Row("速度平滑", &VELOCITY_ALPHA, 0, 1, "%.2f");
        Row("最大速度", &MAX_VELOCITY, 100, 3000, "%.0fpx/s");
        Row("同目标距²", &SAME_TARGET_DIST_SQ, 100, 10000, "%.0f");
        Sep(); CT("offset 衰减");
        Row("offset斜坡", &CFG_OFFSET_RAMP, 5, 200, "%.0fpx");
        Sep(); CT("提示");
        Small("瞄点偏移：-100=框底，0=中心，100=框顶");
        Small("Y偏移：左键触发时叠加到绝对Y坐标");
        Sep(); CT("急停后端状态");
        {
            const char* mm = (g_csBackend == 1) ? "KMBox反向"
                : (g_csBackend == 2) ? "系统反向" : "关闭";
            bool ok = (g_csBackend == 0)
                || (g_csBackend == 2)
                || (g_csBackend == 1 && g_kmConnected);
            ImGui::TextColored(ok ? C(0, .9f, .46f, 1) : C(.9f, .5f, .2f, 1),
                "反向后端: %s", mm);
            if (g_csBackend == 1 && !g_kmConnected)
                ImGui::TextColored(C(.9f, .5f, .2f, 1), "  (KMBox未连接)");
        }
        ImGui::EndChild(); ImGui::PopStyleColor();
    }

    // ============================================================
    //  页：高级
    // ============================================================
    void PageAdv()
    {
        Pad();
        float W = ImGui::GetContentRegionAvail().x;
        float t = (W - 16) / 3.f;

        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##spd", { t,0 }, true);
        CT("限速表（px/帧）");
        Row("近距边界", &CFG_NEAR_DIST, 5, 100, "%.0fpx");
        Row("中距边界", &CFG_MID_DIST, 10, 200, "%.0fpx");
        Row("远距边界", &CFG_FAR_DIST, 20, 300, "%.0fpx");
        ImGui::Spacing();
        RowI("近距限速", &CFG_MAX_NEAR, 1, 20, "%d");
        RowI("中距限速", &CFG_MAX_MID, 1, 50, "%d");
        RowI("远距限速", &CFG_MAX_FFAR, 1, 100, "%d");
        RowI("超远限速", &CFG_MAX_FAR, 1, 200, "%d");
        ImGui::EndChild(); ImGui::PopStyleColor();
        ImGui::SameLine(0, 8);

        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##bez", { t,0 }, true);
        CT("贝塞尔移动");
        ImGui::Checkbox("启用贝塞尔", &CFG_BEZ_ENABLED);
        ImGui::Spacing();
        Row("触发距离", &CFG_BEZ_DIST_THRESH, 10, 300, "%.0fpx");
        Row("曲线偏移", &CFG_BEZ_DEVIATION, 0, 1, "%.2f");
        RowI("最小延迟", &CFG_BEZ_MIN_DELAY, 0, 20, "%dms");
        RowI("最大延迟", &CFG_BEZ_MAX_DELAY, 0, 30, "%dms");
        RowI("动画时长", &CFG_BEZ_DURATION_MS, 5, 50, "%dms");
        RowI("步进时长", &CFG_BEZ_STEP_MS, 1, 20, "%dms");
        RowI("直线阈值", &CFG_BEZ_DIRECT_THRESH, 1, 30, "%dpx");
        ImGui::EndChild(); ImGui::PopStyleColor();
        ImGui::SameLine(0, 8);

        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##osc", { t,0 }, true);
        CT("震荡抑制");
        ImGui::Checkbox("启用震荡抑制", &CFG_OSC_ENABLED);
        ImGui::Spacing();
        Row("衰减系数", &CFG_OSC_DECAY, .01f, 1, "%.2f");
        Row("近距阈值", &CFG_OSC_NEAR_DIST, 10, 200, "%.0fpx");
        RowI("稳定帧数", &CFG_OSC_STABLE_FRAMES, 1, 20, "%d帧");
        Sep(); CT("亚像素");
        Small("亚像素累积器自动管理，无需手动配置");
        ImGui::EndChild(); ImGui::PopStyleColor();
    }

    // ============================================================
    //  页：找色
    // ============================================================
    void PageFindColor()
    {
        Pad();
        float W = ImGui::GetContentRegionAvail().x;
        float h = (W - 12) * .5f;

        // 左：开关 + HSV 范围1
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##fc_l", { h,0 }, true);
        CT("找色设置");

        ImGui::Checkbox("启用找色辅助", &open_findcolor);
        ImGui::Spacing(); Sep();

        CT("HSV 范围 1（红色低端）");
        Row("H1 下限", &fc_h1lo, 0, 180, "%.0f");
        Row("H1 上限", &fc_h1hi, 0, 180, "%.0f");
        Row("S1 下限", &fc_s1lo, 0, 255, "%.0f");
        Row("S1 上限", &fc_s1hi, 0, 255, "%.0f");
        Row("V1 下限", &fc_v1lo, 0, 255, "%.0f");
        Row("V1 上限", &fc_v1hi, 0, 255, "%.0f");
        Sep();
        CT("HSV 范围 2（红色高端）");
        Row("H2 下限", &fc_h2lo, 0, 180, "%.0f");
        Row("H2 上限", &fc_h2hi, 0, 180, "%.0f");
        Row("S2 下限", &fc_s2lo, 0, 255, "%.0f");
        Row("S2 上限", &fc_s2hi, 0, 255, "%.0f");
        Row("V2 下限", &fc_v2lo, 0, 255, "%.0f");
        Row("V2 上限", &fc_v2hi, 0, 255, "%.0f");
        ImGui::EndChild(); ImGui::PopStyleColor();

        ImGui::SameLine(0, 12);

        // 右：截图/瞄准范围
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##fc_r", { h,0 }, true);
        CT("ROI 截图范围（相对截图中心）");
        RowI("ROI X", &fc_roi_x, -400, 400, "%d px");
        RowI("ROI Y", &fc_roi_y, -400, 400, "%d px");
        RowI("ROI 宽", &fc_roi_w, 10, 500, "%d px");
        RowI("ROI 高", &fc_roi_h, 10, 500, "%d px");
        Sep();
        CT("瞄准缩放");
        Row("aim_scale", &fc_aim_scale, .1f, 2.f, "%.2f");
        Small("目标坐标 = 原始坐标 x 缩放系数");
        Sep();
        CT("轮廓过滤");
        RowI("最小面积", &fc_min_area, 1, 500, "%d px2");
        Sep();
        CT("当前 HSV 预览");
        // 色块预览：范围1（红）
        ImDrawList* dl = ImGui::GetWindowDrawList();
        ImVec2 p = ImGui::GetCursorScreenPos();
        float bh = 22.f, bw = (ImGui::GetContentRegionAvail().x - 8) * .5f;
        dl->AddRectFilled(p, { p.x + bw,p.y + bh },
            IM_COL32(220, 40, 40, 200), 4.f);
        dl->AddRectFilled({ p.x + bw + 8,p.y }, { p.x + bw * 2 + 8,p.y + bh },
            IM_COL32(200, 180, 180, 200), 4.f);
        ImGui::Dummy({ W - 20,bh + 4 });
        Small("左=范围1颜色示意  右=范围2颜色示意");
        ImGui::EndChild(); ImGui::PopStyleColor();
    }

    // ============================================================
    //  页：设备
    // ============================================================
    void PageDevice()
    {
        Pad();
        float W = ImGui::GetContentRegionAvail().x;
        float h = (W - 12) * .5f;

        // 左：KMBox
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##km", { h,210 }, true);
        CT("KMBox 设备");
        LabelInput("IP 地址", g_kmIP, 64);
        LabelInput("端口", g_kmPort, 16);
        LabelInput("UUID/MAC", g_kmMAC, 32);
        ImGui::Spacing();
        if (ImGui::Button("重新连接##kmconn", { -1,30 })) {
            // kmNet_init(g_kmIP,g_kmPort,g_kmMAC); kmNet_monitor(1);
        }
        ImGui::EndChild(); ImGui::PopStyleColor();

        ImGui::SameLine(0, 12);

        // 右：引擎/模型
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##eng", { h,210 }, true);
        CT("推理引擎 & 模型");
        const char* engs[] = { "TensorRT (.engine)","ONNXRuntime (.onnx)" };
        LabelCombo("引擎类型", &inference_engine_mode, engs, 2);
        const char* mdls[] = { "YOLOv10 [1,300,6]","YOLOv11 [1,4+nc,a]" };
        LabelCombo("模型版本", &model_way, mdls, 2);
        ImGui::Spacing();
        ImGui::Text("模型路径");
        ImGui::SetNextItemWidth(-116);
        ImGui::InputText("##mp", g_modelPath, 512);
        ImGui::SameLine();
        if (ImGui::Button("浏览##mpbrowse", { 106,0 })) {
            OPENFILENAMEA ofn{};
            char tmp[512] = {};
            strncpy_s(tmp, g_modelPath, 511);
            ofn.lStructSize = sizeof(ofn);
            ofn.lpstrFile = tmp;
            ofn.nMaxFile = sizeof(tmp);
            ofn.lpstrFilter = "模型文件 *.engine;*.onnx 所有文件 *.* ";
            ofn.lpstrTitle = "选择模型文件";
            ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;
            if (GetOpenFileNameA(&ofn))
                strncpy_s(g_modelPath, tmp, 511);
        }
        ImGui::Spacing();
        if (ImGui::Button("应用并重载##reload", { -1,30 })) {
            // 交给主控线程：它会确定性全停→重载引擎→放行。
            // 不在 UI 线程动 g_running，也不 sleep（避免卡 UI）。
            g_requestResume = true;
        }
        ImGui::EndChild(); ImGui::PopStyleColor();

        // 模型输入分辨率
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##mis", { 0,80 }, true);
        CT("模型输入分辨率");
        RowI("模型尺寸", &g_modelInputSize, 32, 1280, "%d px");
        Small("修改后需重载模型才能生效");
        ImGui::EndChild(); ImGui::PopStyleColor();

        // 类别 ID 优先级
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##cat", { 0,160 }, true);
        CT("物体类别优先级 ID");
        ImGui::TextDisabled("按顺序搜索，找到前面的ID后不再搜索后面的");
        ImGui::Spacing();
        ImGui::Text("ID 列表（逗号分隔）：");
        ImGui::SameLine();
        ImGui::SetNextItemWidth(-1);
        if (ImGui::InputText("##catbuf", m_catBuf, sizeof(m_catBuf),
            ImGuiInputTextFlags_EnterReturnsTrue)) {
            ParseCatBuf();
        }
        ImGui::Spacing();
        // 显示当前解析结果
        ImGui::Text("当前优先级：");
        ImGui::SameLine();
        for (int i = 0; i < (int)g_categories.size(); i++) {
            if (i > 0) ImGui::SameLine();
            ImGui::PushStyleColor(ImGuiCol_Button, C(.00f, .40f, .20f, 1));
            char lb[16]; snprintf(lb, 16, "  %d  ##cb%d", g_categories[i], i);
            if (ImGui::SmallButton(lb)) {
                // 点击删除该 ID
                g_categories.erase(g_categories.begin() + i);
                SyncCatBufFromVec();
            }
            ImGui::PopStyleColor();
        }
        ImGui::Spacing();
        if (ImGui::Button("应用##catapply", { 80,26 })) ParseCatBuf();
        ImGui::SameLine();
        ImGui::TextDisabled("回车或点应用生效，点击ID标签可删除");
        ImGui::EndChild(); ImGui::PopStyleColor();

        // 功能开关
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##sw", { 0,0 }, true);
        CT("功能开关");
        float col2 = ImGui::GetContentRegionAvail().x * .5f;
        ImGui::Columns(2, "swc", false);
        ImGui::SetColumnWidth(0, col2);

        static bool useKMBox = true;
        if (ImGui::Checkbox("KMBox 驱动", &useKMBox)) SetMoveMethod(useKMBox ? 0 : 1);
        Small("  关闭则使用 SendInput");
        ImGui::Spacing();
        ImGui::Checkbox("自动开火", &g_autoFireEnabled);
        Small("  进入开火半径后触发");
        ImGui::Spacing();
        ImGui::Checkbox("贝塞尔移动", &CFG_BEZ_ENABLED);
        Small("  远距离曲线移动轨迹");

        ImGui::NextColumn();

        ImGui::Checkbox("震荡抑制", &CFG_OSC_ENABLED);
        Small("  防止鼠标左右抖动");
        ImGui::Spacing();
        bool win = open_test != 0;
        if (ImGui::Checkbox("预览窗口", &win)) open_test = win ? 1 : 0;
        Small("  显示 OpenCV 截图预览");
        ImGui::Spacing();
        ImGui::Checkbox("找色辅助", &open_findcolor);
        Small("  启用颜色辅助瞄准");

        ImGui::Columns(1);
        ImGui::EndChild(); ImGui::PopStyleColor();
    }

    // ============================================================
    //  页：配置
    // ============================================================
    void PageConfig()
    {
        Pad();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild("##cfg", { 0,0 }, true);
        CT("配置文件管理");
        ImGui::TextDisabled("配置保存到 aim_config.ini，与 exe 同目录");
        ImGui::Spacing();

        ImGui::PushStyleColor(ImGuiCol_Button, C(.00f, .45f, .23f, 1));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, C(.00f, .60f, .30f, 1));
        if (ImGui::Button("保存配置##save", { 160,36 })) DoSave();
        ImGui::PopStyleColor(2);

        ImGui::SameLine(0, 12);

        ImGui::PushStyleColor(ImGuiCol_Button, C(.20f, .25f, .30f, 1));
        if (ImGui::Button("加载配置##load", { 160,36 })) DoLoad();
        ImGui::PopStyleColor();

        ImGui::SameLine(0, 12);

        ImGui::PushStyleColor(ImGuiCol_Button, C(.40f, .12f, .05f, 1));
        if (ImGui::Button("重置默认##reset", { 160,36 })) {
            CFG_KP = .20f; CFG_KI = .018f; CFG_KD = .002f;
            y_offset = 12; conf = .25f; g_fireRadius = 3.f;
            snprintf(m_saveMsg, sizeof(m_saveMsg), "已重置为默认值");
            m_saveMsgTimer = 3.f;
        }
        ImGui::PopStyleColor();

        ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing();
        if (m_saveMsgTimer > 0.f) {
            ImGui::TextColored(C(0, .9f, .46f, 1), "%s", m_saveMsg);
            ImGui::Spacing();
        }

        CT("当前参数预览");
        ImGui::Text("Kp=%.3f  Ki=%.3f  Kd=%.3f", CFG_KP, CFG_KI, CFG_KD);
        ImGui::Text("置信度=%.2f  截图半径=%d  开火半径=%.1f", conf, g_captureRadius, g_fireRadius);
        ImGui::Text("类别优先级：");
        ImGui::SameLine();
        for (int i = 0; i < (int)g_categories.size(); i++) {
            if (i > 0) { ImGui::SameLine(); ImGui::TextDisabled(" > "); }
            ImGui::SameLine();
            char tb[16]; snprintf(tb, 16, "%d", g_categories[i]);
            ImGui::TextColored(C(0, .9f, .46f, 1), "%s", tb);
        }
        ImGui::Spacing();
        ImGui::Text("找色：%s  HSV1=[%.0f-%.0f / %.0f-%.0f / %.0f-%.0f]",
            open_findcolor ? "ON" : "OFF",
            fc_h1lo, fc_h1hi, fc_s1lo, fc_s1hi, fc_v1lo, fc_v1hi);

        ImGui::EndChild(); ImGui::PopStyleColor();
    }

    // ── 配置保存/加载 ─────────────────────────────────────────
    void DoSave()
    {
        std::ofstream f("aim_config.ini");
        if (!f) { snprintf(m_saveMsg, sizeof(m_saveMsg), "保存失败！"); m_saveMsgTimer = 3.f; return; }
        f << "kp=" << CFG_KP << "\nki=" << CFG_KI << "\nkd=" << CFG_KD << "\n";
        f << "pred_wx=" << CFG_PRED_WEIGHT_X << "\npred_wy=" << CFG_PRED_WEIGHT_Y << "\n";
        f << "pred_zd=" << CFG_PRED_ZERO_DIST << "\npred_fd=" << CFG_PRED_FULL_DIST << "\n";
        f << "out_max=" << CFG_OUT_MAX << "\nramp=" << CFG_RAMP_MS << "\ninit_scale=" << CFG_INIT_SCALE << "\n";
        f << "dz=" << CFG_DZ_DIST << "\niclear=" << CFG_INTEGRAL_CLEAR_DIST << "\n";
        f << "filter_alpha=" << CFG_FILTER_ALPHA_NEAR << "\nfilter_nd=" << CFG_FILTER_NEAR_DIST << "\n";
        f << "osc_decay=" << CFG_OSC_DECAY << "\nosc_nd=" << CFG_OSC_NEAR_DIST << "\nosc_sf=" << CFG_OSC_STABLE_FRAMES << "\n";
        f << "near=" << CFG_NEAR_DIST << "\nmid=" << CFG_MID_DIST << "\nfar=" << CFG_FAR_DIST << "\n";
        f << "maxN=" << CFG_MAX_NEAR << "\nmaxM=" << CFG_MAX_MID << "\nmaxFF=" << CFG_MAX_FFAR << "\nmaxF=" << CFG_MAX_FAR << "\n";
        f << "bez_thresh=" << CFG_BEZ_DIST_THRESH << "\nbez_dev=" << CFG_BEZ_DEVIATION << "\n";
        f << "y_off=" << y_offset << "\nmax_lost=" << MAX_LOST_FRAMES << "\npredict_t=" << PREDICT_TIME << "\n";
        f << "vel_alpha=" << VELOCITY_ALPHA << "\nmax_vel=" << MAX_VELOCITY << "\nsame_sq=" << SAME_TARGET_DIST_SQ << "\n";
        f << "conf=" << conf << "\nauto_fire=" << g_autoFireEnabled << "\nfire_r=" << g_fireRadius << "\n";
        f << "aim_pct=" << CFG_AIM_OFFSET_PCT << "\ncap_r=" << g_captureRadius << "\n";
        f << "cs_backend=" << g_csBackend << "\ncs=" << (g_counterStrafing ? 1 : 0) << "\n";
        f << "findcolor=" << open_findcolor << "\n";
        f << "fc_h1lo=" << fc_h1lo << "\nfc_h1hi=" << fc_h1hi << "\n";
        f << "fc_s1lo=" << fc_s1lo << "\nfc_s1hi=" << fc_s1hi << "\n";
        f << "fc_v1lo=" << fc_v1lo << "\nfc_v1hi=" << fc_v1hi << "\n";
        f << "fc_h2lo=" << fc_h2lo << "\nfc_h2hi=" << fc_h2hi << "\n";
        f << "fc_s2lo=" << fc_s2lo << "\nfc_s2hi=" << fc_s2hi << "\n";
        f << "fc_v2lo=" << fc_v2lo << "\nfc_v2hi=" << fc_v2hi << "\n";
        f << "fc_roi_x=" << fc_roi_x << "\nfc_roi_y=" << fc_roi_y << "\n";
        f << "fc_roi_w=" << fc_roi_w << "\nfc_roi_h=" << fc_roi_h << "\n";
        f << "fc_aim_scale=" << fc_aim_scale << "\nfc_min_area=" << fc_min_area << "\n";
        // 类别列表
        f << "categories=";
        for (int i = 0; i < (int)g_categories.size(); i++) {
            if (i) f << ","; f << g_categories[i];
        }
        f << "\n";
        snprintf(m_saveMsg, sizeof(m_saveMsg), "已保存到 aim_config.ini");
        m_saveMsgTimer = 3.f;
    }

    void DoLoad()
    {
        std::ifstream f("aim_config.ini");
        if (!f) { snprintf(m_saveMsg, sizeof(m_saveMsg), "找不到配置文件！"); m_saveMsgTimer = 3.f; return; }
        std::string line;
        while (std::getline(f, line)) {
            auto eq = line.find('=');
            if (eq == std::string::npos) continue;
            std::string k = line.substr(0, eq), v = line.substr(eq + 1);
            try {
                if (k == "kp") CFG_KP = std::stof(v);
                else if (k == "ki") CFG_KI = std::stof(v);
                else if (k == "kd") CFG_KD = std::stof(v);
                else if (k == "pred_wx") CFG_PRED_WEIGHT_X = std::stof(v);
                else if (k == "pred_wy") CFG_PRED_WEIGHT_Y = std::stof(v);
                else if (k == "pred_zd") CFG_PRED_ZERO_DIST = std::stof(v);
                else if (k == "pred_fd") CFG_PRED_FULL_DIST = std::stof(v);
                else if (k == "out_max") CFG_OUT_MAX = std::stof(v);
                else if (k == "ramp")    CFG_RAMP_MS = std::stof(v);
                else if (k == "init_scale") CFG_INIT_SCALE = std::stof(v);
                else if (k == "dz")      CFG_DZ_DIST = std::stof(v);
                else if (k == "iclear")  CFG_INTEGRAL_CLEAR_DIST = std::stof(v);
                else if (k == "filter_alpha") CFG_FILTER_ALPHA_NEAR = std::stof(v);
                else if (k == "filter_nd")    CFG_FILTER_NEAR_DIST = std::stof(v);
                else if (k == "osc_decay") CFG_OSC_DECAY = std::stof(v);
                else if (k == "osc_nd")    CFG_OSC_NEAR_DIST = std::stof(v);
                else if (k == "osc_sf")    CFG_OSC_STABLE_FRAMES = std::stoi(v);
                else if (k == "near") CFG_NEAR_DIST = std::stof(v);
                else if (k == "mid")  CFG_MID_DIST = std::stof(v);
                else if (k == "far")  CFG_FAR_DIST = std::stof(v);
                else if (k == "maxN") CFG_MAX_NEAR = std::stoi(v);
                else if (k == "maxM") CFG_MAX_MID = std::stoi(v);
                else if (k == "maxFF")CFG_MAX_FFAR = std::stoi(v);
                else if (k == "maxF") CFG_MAX_FAR = std::stoi(v);
                else if (k == "bez_thresh") CFG_BEZ_DIST_THRESH = std::stof(v);
                else if (k == "bez_dev")    CFG_BEZ_DEVIATION = std::stof(v);
                else if (k == "y_off")     y_offset = std::stoi(v);
                else if (k == "max_lost")  MAX_LOST_FRAMES = std::stoi(v);
                else if (k == "predict_t") PREDICT_TIME = std::stof(v);
                else if (k == "vel_alpha") VELOCITY_ALPHA = std::stof(v);
                else if (k == "max_vel")   MAX_VELOCITY = std::stof(v);
                else if (k == "same_sq")   SAME_TARGET_DIST_SQ = std::stof(v);
                else if (k == "conf")      conf = std::stof(v);
                else if (k == "auto_fire") g_autoFireEnabled = v == "1" || v == "true";
                else if (k == "fire_r")    g_fireRadius = std::stof(v);
                else if (k == "aim_pct")   CFG_AIM_OFFSET_PCT = std::stof(v);
                else if (k == "cap_r")     g_captureRadius = std::stoi(v);
                else if (k == "cs_backend") g_csBackend = std::clamp(std::stoi(v), 0, 2);
                else if (k == "cs")        g_counterStrafing = (v == "1" || v == "true");
                else if (k == "findcolor") open_findcolor = v == "1" || v == "true";
                else if (k == "fc_h1lo") fc_h1lo = std::stof(v);
                else if (k == "fc_h1hi") fc_h1hi = std::stof(v);
                else if (k == "fc_s1lo") fc_s1lo = std::stof(v);
                else if (k == "fc_s1hi") fc_s1hi = std::stof(v);
                else if (k == "fc_v1lo") fc_v1lo = std::stof(v);
                else if (k == "fc_v1hi") fc_v1hi = std::stof(v);
                else if (k == "fc_h2lo") fc_h2lo = std::stof(v);
                else if (k == "fc_h2hi") fc_h2hi = std::stof(v);
                else if (k == "fc_s2lo") fc_s2lo = std::stof(v);
                else if (k == "fc_s2hi") fc_s2hi = std::stof(v);
                else if (k == "fc_v2lo") fc_v2lo = std::stof(v);
                else if (k == "fc_v2hi") fc_v2hi = std::stof(v);
                else if (k == "fc_roi_x") fc_roi_x = std::stoi(v);
                else if (k == "fc_roi_y") fc_roi_y = std::stoi(v);
                else if (k == "fc_roi_w") fc_roi_w = std::stoi(v);
                else if (k == "fc_roi_h") fc_roi_h = std::stoi(v);
                else if (k == "fc_aim_scale") fc_aim_scale = std::stof(v);
                else if (k == "fc_min_area")  fc_min_area = std::stoi(v);
                else if (k == "categories") {
                    g_categories.clear();
                    std::istringstream ss(v);
                    std::string tok;
                    while (std::getline(ss, tok, ',')) {
                        if (!tok.empty()) g_categories.push_back(std::stoi(tok));
                    }
                    SyncCatBufFromVec();
                }
            }
            catch (...) {}
        }
        snprintf(m_saveMsg, sizeof(m_saveMsg), "配置加载成功！");
        m_saveMsgTimer = 3.f;
    }

    // ── 类别 buf 同步 ─────────────────────────────────────────
    void SyncCatBufFromVec() {
        m_catBuf[0] = '\0';
        for (int i = 0; i < (int)g_categories.size(); i++) {
            char tmp[12]; snprintf(tmp, 12, i ? ", %d" : "%d", g_categories[i]);
            strncat(m_catBuf, tmp, sizeof(m_catBuf) - strlen(m_catBuf) - 1);
        }
    }

    void ParseCatBuf() {
        g_categories.clear();
        std::istringstream ss(m_catBuf);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            // 去空格
            tok.erase(remove_if(tok.begin(), tok.end(), ::isspace), tok.end());
            if (!tok.empty()) {
                try { g_categories.push_back(std::stoi(tok)); }
                catch (...) {}
            }
        }
        // 去重但保持顺序
        std::vector<int> seen;
        auto it = std::remove_if(g_categories.begin(), g_categories.end(), [&](int x) {
            if (std::find(seen.begin(), seen.end(), x) != seen.end()) return true;
            seen.push_back(x); return false;
            });
        g_categories.erase(it, g_categories.end());
        SyncCatBufFromVec();
    }

    // ── 帮助 ─────────────────────────────────────────────────
    static ImVec4 C(float r, float g, float b, float a) { return{ r,g,b,a }; }
    void Pad() { ImGui::SetCursorPosX(16); ImGui::SetCursorPosY(ImGui::GetCursorPosY() + 12); }
    void Sep() { ImGui::Spacing(); ImGui::Separator(); ImGui::Spacing(); }
    void CT(const char* t) {
        ImGui::PushStyleColor(ImGuiCol_Text, C(.5f, .52f, .55f, 1));
        ImGui::SetWindowFontScale(.85f); ImGui::Text("%s", t);
        ImGui::SetWindowFontScale(1.f); ImGui::PopStyleColor();
        ImGui::Spacing();
    }
    void Small(const char* t) {
        ImGui::PushStyleColor(ImGuiCol_Text, C(.5f, .52f, .55f, 1));
        ImGui::SetWindowFontScale(.85f); ImGui::TextWrapped("%s", t);
        ImGui::SetWindowFontScale(1.f); ImGui::PopStyleColor();
    }

    void Row(const char* lbl, float* v, float lo, float hi, const char* fmt) {
        float av = ImGui::GetContentRegionAvail().x;
        ImGui::PushStyleColor(ImGuiCol_Text, C(.75f, .78f, .82f, 1));
        ImGui::Text("%s", lbl); ImGui::PopStyleColor();
        // 滑条
        ImGui::SameLine(130); ImGui::SetNextItemWidth(av - 290);
        std::string sid = std::string("##f") + lbl;
        ImGui::SliderFloat(sid.c_str(), v, lo, hi, "");
        // 输入框（可直接键入）
        ImGui::SameLine();
        ImGui::SetNextItemWidth(88);
        std::string iid = std::string("##fi") + lbl;
        ImGui::InputFloat(iid.c_str(), v, 0, 0, fmt,
            ImGuiInputTextFlags_CharsDecimal);
        *v = std::clamp(*v, lo, hi);
        ImGui::Spacing();
    }
    // 返回值：本帧是否“结束编辑”（拖完滑块或输入框松手）。旧调用忽略返回值即可。
    bool RowI(const char* lbl, int* v, int lo, int hi, const char* fmt) {
        float av = ImGui::GetContentRegionAvail().x;
        ImGui::PushStyleColor(ImGuiCol_Text, C(.75f, .78f, .82f, 1));
        ImGui::Text("%s", lbl); ImGui::PopStyleColor();
        ImGui::SameLine(130); ImGui::SetNextItemWidth(av - 290);
        std::string sid = std::string("##i") + lbl;
        ImGui::SliderInt(sid.c_str(), v, lo, hi, "");
        bool done = ImGui::IsItemDeactivatedAfterEdit();
        ImGui::SameLine();
        ImGui::SetNextItemWidth(88);
        std::string iid = std::string("##ii") + lbl;
        ImGui::InputInt(iid.c_str(), v, 0, 0);
        done = done || ImGui::IsItemDeactivatedAfterEdit();
        *v = std::clamp(*v, lo, hi);
        ImGui::Spacing();
        return done;
    }
    void LabelInput(const char* lbl, char* buf, int sz) {
        ImGui::Text("%s", lbl); ImGui::SameLine(110);
        ImGui::SetNextItemWidth(-1);
        std::string id = std::string("##li") + lbl;
        ImGui::InputText(id.c_str(), buf, sz);
        ImGui::Spacing();
    }
    void LabelCombo(const char* lbl, int* v, const char** items, int n) {
        ImGui::Text("%s", lbl); ImGui::SameLine(110);
        ImGui::SetNextItemWidth(-1);
        std::string id = std::string("##cm") + lbl;
        ImGui::Combo(id.c_str(), v, items, n);
        ImGui::Spacing();
    }
    void ChartBox(const char* id, const char* title,
        const std::deque<float>& q, float lo, float hi,
        float w, float h, ImVec4 col) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild(id, { w,h + 30 }, true);
        Small(title);
        if (!q.empty()) {
            std::vector<float> v(q.begin(), q.end());
            ImGui::PushStyleColor(ImGuiCol_PlotLines, col);
            ImGui::PlotLines(std::string(std::string("##pl") + id).c_str(),
                v.data(), (int)v.size(), 0, nullptr, lo, hi, { -1,ImGui::GetContentRegionAvail().y - 4 });
            ImGui::PopStyleColor();
        }
        ImGui::EndChild(); ImGui::PopStyleColor();
    }
    void Card(const char* id, const char* lbl,
        const char* big, ImVec4 bigC, const char* sub) {
        ImGui::PushStyleColor(ImGuiCol_ChildBg, C(.10f, .11f, .13f, 1));
        ImGui::BeginChild(id, { 0,90 }, true);
        Small(lbl);
        ImGui::TextColored(bigC, "%s", big);
        Small(sub);
        ImGui::EndChild(); ImGui::PopStyleColor();
    }
    void PushHist(std::deque<float>& q, float v) {
        q.push_back(v);
        if ((int)q.size() > HIST) q.pop_front();
    }

    // ── 主题 ─────────────────────────────────────────────────
    static void Theme() {
        ImGui::StyleColorsDark();
        ImGuiStyle& s = ImGui::GetStyle();
        s.WindowRounding = 0; s.ChildRounding = 8; s.FrameRounding = 5;
        s.GrabRounding = 5; s.TabRounding = 6; s.ScrollbarRounding = 6;
        s.WindowBorderSize = 0; s.ChildBorderSize = .5f;
        s.FramePadding = { 8,5 }; s.ItemSpacing = { 8,8 }; s.GrabMinSize = 12;
        ImVec4* c = s.Colors;
        c[ImGuiCol_WindowBg] = { .08f,.09f,.10f,1 };
        c[ImGuiCol_ChildBg] = { .10f,.11f,.13f,1 };
        c[ImGuiCol_FrameBg] = { .14f,.15f,.17f,1 };
        c[ImGuiCol_FrameBgHovered] = { .18f,.20f,.22f,1 };
        c[ImGuiCol_FrameBgActive] = { .20f,.22f,.25f,1 };
        c[ImGuiCol_SliderGrab] = { .00f,.90f,.46f,1 };
        c[ImGuiCol_SliderGrabActive] = { .00f,.70f,.36f,1 };
        c[ImGuiCol_CheckMark] = { .00f,.90f,.46f,1 };
        c[ImGuiCol_Button] = { .14f,.16f,.18f,1 };
        c[ImGuiCol_ButtonHovered] = { .20f,.22f,.26f,1 };
        c[ImGuiCol_ButtonActive] = { .00f,.50f,.25f,1 };
        c[ImGuiCol_Header] = { .14f,.15f,.17f,1 };
        c[ImGuiCol_HeaderHovered] = { .18f,.20f,.22f,1 };
        c[ImGuiCol_Separator] = { .18f,.20f,.22f,1 };
        c[ImGuiCol_Text] = { .91f,.92f,.93f,1 };
        c[ImGuiCol_TextDisabled] = { .50f,.52f,.55f,1 };
        c[ImGuiCol_PlotLines] = { .00f,.90f,.46f,1 };
        c[ImGuiCol_PlotHistogram] = { .00f,.90f,.46f,1 };
        c[ImGuiCol_PopupBg] = { .12f,.13f,.15f,.98f };
        c[ImGuiCol_Tab] = { .10f,.11f,.13f,1 };
        c[ImGuiCol_TabHovered] = { .14f,.16f,.18f,1 };
        c[ImGuiCol_TabActive] = { .00f,.55f,.28f,1 };
        c[ImGuiCol_TabUnfocused] = { .10f,.11f,.13f,1 };
        c[ImGuiCol_TabUnfocusedActive] = { .00f,.45f,.23f,1 };
    }

    // ── DX11 ─────────────────────────────────────────────────
    bool InitDX11() {
        DXGI_SWAP_CHAIN_DESC sd{};
        sd.BufferCount = 2;
        sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
        sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
        sd.OutputWindow = m_hwnd;
        sd.SampleDesc.Count = 1;
        sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
        sd.Windowed = TRUE;
        D3D_FEATURE_LEVEL fl;
        return SUCCEEDED(D3D11CreateDeviceAndSwapChain(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0,
            nullptr, 0, D3D11_SDK_VERSION, &sd, &m_sc, &m_dev, &fl, &m_ctx))
            && MakeRTV();
    }
    bool MakeRTV() {
        ID3D11Texture2D* buf = nullptr;
        m_sc->GetBuffer(0, IID_PPV_ARGS(&buf));
        if (!buf) return false;
        m_dev->CreateRenderTargetView(buf, nullptr, &m_rtv);
        buf->Release(); return m_rtv != nullptr;
    }
    void FreeDX11() {
        if (m_rtv) { m_rtv->Release(); m_rtv = nullptr; }
        if (m_sc) { m_sc->Release(); m_sc = nullptr; }
        if (m_ctx) { m_ctx->Release(); m_ctx = nullptr; }
        if (m_dev) { m_dev->Release(); m_dev = nullptr; }
    }

    // ── WndProc ──────────────────────────────────────────────
    static LRESULT CALLBACK WndProc(HWND h, UINT m, WPARAM w, LPARAM l) {
        if (ImGui_ImplWin32_WndProcHandler(h, m, w, l)) return true;
        switch (m) {
        case WM_CREATE: {
            auto cs = (LPCREATESTRUCT)l;
            SetWindowLongPtrW(h, GWLP_USERDATA, (LONG_PTR)cs->lpCreateParams);
            break;
        }
        case WM_SIZE:
            if (w != SIZE_MINIMIZED) {
                auto* ui = (ImGuiUI*)GetWindowLongPtrW(h, GWLP_USERDATA);
                if (ui && ui->m_rtv) {
                    ui->m_rtv->Release(); ui->m_rtv = nullptr;
                    ui->m_sc->ResizeBuffers(0, LOWORD(l), HIWORD(l), DXGI_FORMAT_UNKNOWN, 0);
                    ui->MakeRTV();
                }
            } return 0;
        case WM_SYSCOMMAND:
            if ((w & 0xfff0) == SC_KEYMENU) return 0; break;
        case WM_DESTROY:
            PostQuitMessage(0); return 0;
        }
        return DefWindowProcW(h, m, w, l);
    }
};
