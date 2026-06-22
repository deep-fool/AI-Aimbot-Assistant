#define NOMINMAX
#include "Detector.h"
#include <Windows.h>
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <thread>
#include <cstring>

// 记录单帧"纯推理"耗时（不含预处理/后处理），单位 ms
static thread_local double g_lastPureInferMs = 0.0;

Detector::Detector(const std::string& modelPath, int modelWay, InferenceEngine engineType)
    : model_way(modelWay), inference_engine(engineType)
{
    buffers[0] = nullptr;
    buffers[1] = nullptr;

    if (inference_engine == InferenceEngine::TensorRT) {
        InitTensorRT(modelPath);
    }
    else {
        InitONNXRuntime(modelPath);
    }
}

Detector::~Detector()
{
    FreeBuffers();
    if (inference_engine == InferenceEngine::TensorRT) {
        cudaStreamDestroy(stream);
        if (context) delete context;
        if (engine)  delete engine;
        if (runtime) delete runtime;
    }
}

// =============================================================
// TensorRT 初始化
// =============================================================
void Detector::InitTensorRT(const std::string& modelPath)
{
    std::ifstream file(modelPath, std::ios::binary);
    if (!file)
        throw std::runtime_error("engine 文件无法打开: " + modelPath);

    file.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(file.tellg());
    file.seekg(0, std::ios::beg);

    std::vector<char> data(size);
    file.read(data.data(), static_cast<std::streamsize>(size));

    runtime = nvinfer1::createInferRuntime(gLogger);
    if (!runtime)
        throw std::runtime_error("createInferRuntime 失败");

    engine = runtime->deserializeCudaEngine(data.data(), size);
    if (!engine)
        throw std::runtime_error("deserializeCudaEngine 失败");

    // ---- 打印所有 Tensor，并从输出形状推导 numClasses / numAnchors ----
    printf("[DBG] engine tensors:\n");
    for (int i = 0; i < engine->getNbIOTensors(); i++)
    {
        const char* name = engine->getIOTensorName(i);
        auto        shape = engine->getTensorShape(name);
        auto        mode = engine->getTensorIOMode(name);

        printf("  [%d] %-12s  mode=%s  shape=[", i, name,
            mode == nvinfer1::TensorIOMode::kINPUT ? "INPUT " :
            mode == nvinfer1::TensorIOMode::kOUTPUT ? "OUTPUT" : "NONE  ");
        for (int j = 0; j < shape.nbDims; j++)
            printf("%d%s", (int)shape.d[j], j < shape.nbDims - 1 ? "," : "");
        printf("]\n");

        if (mode == nvinfer1::TensorIOMode::kOUTPUT && shape.nbDims == 3)
        {
            // YOLOv11: [1, 4+nc, 8400]  → d[1]=4+nc, d[2]=anchors
            // YOLOv10: [1, 300,  6   ]  → d[1]=300,  d[2]=6
            int d1 = (int)shape.d[1];
            int d2 = (int)shape.d[2];

            if (d2 == 6) {
                // v10 layout: [1, num_boxes, 6]
                printf("[DBG] Detected YOLOv10 output layout: %d boxes x 6\n", d1);
            }
            else {
                // v11 layout: [1, 4+nc, anchors]
                int nc = d1 - 4;
                if (nc > 0) {
                    numClasses = nc;
                    numAnchors_ = d2;
                    printf("[DBG] Detected YOLOv11 output layout: nc=%d  anchors=%d\n", nc, d2);
                }
            }
        }
    }
    printf("[DBG] numClasses=%d  numAnchors=%d\n", numClasses, numAnchors_);

    context = engine->createExecutionContext();
    if (!context)
        throw std::runtime_error("createExecutionContext 失败");

    cudaStreamCreate(&stream);
}

// =============================================================
// ONNXRuntime 初始化
// =============================================================
void Detector::InitONNXRuntime(const std::string& modelPath)
{
    ortEnv = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DetectorONNX");

    Ort::SessionOptions sessionOptions;
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // ===== DirectML / CUDA EP 动态加载 (优先 DirectML: NVIDIA/AMD/Intel 通用) =====
    {
        // 获取原始 C API 指针 (用于错误处理, Ort::GetApi() 在某些版本不可靠)
        const OrtApi* api = OrtGetApiBase()->GetApi(1);  // API v1 所有版本兼容

        HMODULE ortDll = GetModuleHandleA("onnxruntime.dll");
        bool gpuOK = false;
        const char* deviceName = "CPU";

        if (ortDll) {
            typedef OrtStatus* (ORT_API_CALL* EpAppendFn)(OrtSessionOptions*, int);

            // 1) DirectML (优先 — 支持 NVIDIA/AMD/Intel 显卡)
            //    m_onnxDevicePref: 0=Auto 1=DirectML 2=CUDA 3=CPU
            if (m_onnxDevicePref == 0 || m_onnxDevicePref == 1) {
                EpAppendFn epFn = (EpAppendFn)GetProcAddress(ortDll,
                    "OrtSessionOptionsAppendExecutionProvider_DML");
                if (!epFn) epFn = (EpAppendFn)GetProcAddress(ortDll,
                    "OrtSessionOptionsAppendExecutionProvider_DML1");
                if (epFn) {
                    OrtStatus* st = epFn(sessionOptions, 0);
                    if (!st) {
                        gpuOK = true; deviceName = "DirectML";
                        printf("[DBG] ONNXRuntime: DirectML EP loaded (NVIDIA/AMD/Intel)\n");
                    }
                    else {
                        printf("[DBG] ONNXRuntime: DirectML failed: %s\n",
                            api->GetErrorMessage(st));
                        api->ReleaseStatus(st);
                    }
                }
                else {
                    printf("[DBG] ONNXRuntime: DirectML not in this onnxruntime build\n");
                }
            }

            // 2) CUDA 回退 (仅 NVIDIA)
            if (!gpuOK && (m_onnxDevicePref == 0 || m_onnxDevicePref == 2)) {
                EpAppendFn epFn = (EpAppendFn)GetProcAddress(ortDll,
                    "OrtSessionOptionsAppendExecutionProvider_CUDA");
                if (!epFn) epFn = (EpAppendFn)GetProcAddress(ortDll,
                    "OrtSessionOptionsAppendExecutionProvider_CUDA_V2");
                if (epFn) {
                    OrtStatus* st = epFn(sessionOptions, 0);
                    if (!st) {
                        gpuOK = true; deviceName = "CUDA";
                        printf("[DBG] ONNXRuntime: CUDA EP loaded (NVIDIA)\n");
                    }
                    else {
                        printf("[DBG] ONNXRuntime: CUDA failed: %s\n",
                            api->GetErrorMessage(st));
                        api->ReleaseStatus(st);
                    }
                }
                else {
                    printf("[DBG] ONNXRuntime: CUDA not available\n");
                }
            }
        }

        if (!gpuOK) {
            printf("[DBG] ONNXRuntime: Using CPU inference\n");
        }

        // ===== 性能优化: 内存模式 + 线程配置 =====
        sessionOptions.EnableMemPattern();
        {
            int nThreads = (int)std::thread::hardware_concurrency();
            if (nThreads < 2) nThreads = 2;
            if (nThreads > 8) nThreads = 8;
            sessionOptions.SetIntraOpNumThreads(nThreads);
            sessionOptions.SetInterOpNumThreads(2);
            printf("[DBG] ONNXRuntime: device=%s threads=%d memPattern=ON\n",
                deviceName, nThreads);
        }
    }

#ifdef _WIN32
    std::wstring wpath(modelPath.begin(), modelPath.end());
    ortSession = Ort::Session(ortEnv, wpath.c_str(), sessionOptions);
#else
    ortSession = Ort::Session(ortEnv, modelPath.c_str(), sessionOptions);
#endif

    ortMemInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // ---- 获取输入/输出名 ----
    Ort::AllocatorWithDefaultOptions allocator;
    {
        auto inputNameAlloc = ortSession.GetInputNameAllocated(0, allocator);
        ortInputName = inputNameAlloc.get();
        auto outputNameAlloc = ortSession.GetOutputNameAllocated(0, allocator);
        ortOutputName = outputNameAlloc.get();
    }

    // ---- 打印输入/输出形状，并从输出形状推导 numClasses / numAnchors ----
    printf("[DBG] onnx model: input=%s  output=%s\n", ortInputName.c_str(), ortOutputName.c_str());

    auto outShape = ortSession.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    printf("[DBG] onnx output shape=[");
    for (size_t j = 0; j < outShape.size(); j++)
        printf("%lld%s", (long long)outShape[j], j < outShape.size() - 1 ? "," : "");
    printf("]\n");

    if (outShape.size() == 3)
    {
        int d1 = (int)outShape[1];
        int d2 = (int)outShape[2];

        if (d2 == 6) {
            // v10 layout: [1, num_boxes, 6]
            printf("[DBG] Detected YOLOv10 output layout: %d boxes x 6\n", d1);
        }
        else {
            // v11 layout: [1, 4+nc, anchors]
            int nc = d1 - 4;
            if (nc > 0) {
                numClasses = nc;
                numAnchors_ = d2;
                printf("[DBG] Detected YOLOv11 output layout: nc=%d  anchors=%d\n", nc, d2);
            }
        }
    }
    printf("[DBG] numClasses=%d  numAnchors=%d\n", numClasses, numAnchors_);

    // ===== 快速路径初始化: 预分配缓冲 + 预创建Tensor + LUT + 预热 =====
    BuildLUT();

    // 预分配输入缓冲 (planar float: 3 * inputSize^2)
    const size_t planeSize = (size_t)inputSize * inputSize;
    m_onnxInputData.resize(3 * planeSize);

    // 预构建输入 shape
    m_onnxInputShape = { 1, 3, (int64_t)inputSize, (int64_t)inputSize };

    // 预创建 Ort::Value 包装 m_onnxInputData (零拷贝 — 数据直接写入 tensor 内存)
    m_onnxInputTensor = Ort::Value::CreateTensor<float>(
        ortMemInfo,
        m_onnxInputData.data(),
        m_onnxInputData.size(),
        m_onnxInputShape.data(),
        m_onnxInputShape.size());

    // 预创建 RunOptions (每帧复用, 避免堆分配)
    m_onnxRunOptions = Ort::RunOptions{};

    // 缓存 name 指针 (避免每帧 c_str() 调用)
    m_pInputName = ortInputName.c_str();
    m_pOutputName = ortOutputName.c_str();

    // 模型预热: 首帧推理包含 kernel 编译/内存分配, 提前跑一次空帧
    printf("[DBG] ONNXRuntime: warming up (first inference always slow)...\n");
    {
        std::fill(m_onnxInputData.begin(), m_onnxInputData.end(), 0.0f);
        const auto warm0 = std::chrono::high_resolution_clock::now();
        try {
            std::vector<Ort::Value> warmOutput;
            warmOutput = ortSession.Run(m_onnxRunOptions,
                &m_pInputName, &m_onnxInputTensor, 1,
                &m_pOutputName, 1);
        }
        catch (const Ort::Exception&) {
            // 预热失败非致命 — 首帧真实推理时完成 kernel 编译
            printf("[DBG] ONNXRuntime: warmup failed (non-fatal)\n");
        }
        const auto warm1 = std::chrono::high_resolution_clock::now();
        double warmMs = std::chrono::duration<double, std::milli>(warm1 - warm0).count();
        printf("[DBG] ONNXRuntime: warmup %.1f ms\n", warmMs);
    }
}

// =============================================================
// SetInputSize
// =============================================================
void Detector::SetInputSize(int size)
{
    if (size <= 0)
        throw std::invalid_argument("inputSize 必须大于 0");
    if (buffersAllocated)
        FreeBuffers();
    inputSize = size;

    // 公共: 分配 FastPreprocess 用的 CPU 缓冲 + 初始化 LUT (TensorRT & ONNX 都需要)
    static bool s_lutBuilt = false;
    if (!s_lutBuilt) { BuildLUT(); s_lutBuilt = true; }

    const size_t planeFloats = (size_t)inputSize * inputSize;
    m_onnxInputData.clear();
    m_onnxInputData.resize(3 * planeFloats);
    m_lastSrcW = 0;
    m_lastSrcH = 0;

    if (inference_engine == InferenceEngine::TensorRT) {
        AllocateBuffers();
    }
    else {
        // ONNX 特有: 预创建 tensor + RunOptions
        m_onnxInputShape = { 1, 3, (int64_t)inputSize, (int64_t)inputSize };

        m_onnxInputTensor = Ort::Value{ nullptr };
        m_onnxInputTensor = Ort::Value::CreateTensor<float>(
            ortMemInfo,
            m_onnxInputData.data(),
            m_onnxInputData.size(),
            m_onnxInputShape.data(),
            m_onnxInputShape.size());

        buffersAllocated = true;
    }
}

// =============================================================
// AllocateBuffers / FreeBuffers (仅 TensorRT 使用)
// =============================================================
void Detector::AllocateBuffers()
{
    const size_t inFloats = static_cast<size_t>(3) * inputSize * inputSize;
    cudaMalloc(&buffers[0], inFloats * sizeof(float));

    size_t outFloats;
    if (model_way == 0) {
        // YOLOv10: 输出 [1, 300, 6]，固定 300 个候选框
        outFloats = static_cast<size_t>(300) * 6;
        cudaMalloc(&buffers[1], outFloats * sizeof(float));
        printf("[DBG] AllocateBuffers(v10): inputSize=%d  boxes=300 stride=6\n", inputSize);
    }
    else {
        // YOLOv11: 按实际 anchor 数分配
        const int rowDim = 4 + numClasses;
        outFloats = static_cast<size_t>(rowDim) * numAnchors_;
        cudaMalloc(&buffers[1], outFloats * sizeof(float));
        printf("[DBG] AllocateBuffers(v11): inputSize=%d  rowDim=%d  numAnchors=%d\n",
            inputSize, rowDim, numAnchors_);
    }

    // ── 页锁(pinned)主机缓冲：让 H2D / D2H 走真正的异步 DMA，而非 pageable 慢路径 ──
    //   这是 TensorRT 比 DML 慢的主因：原先用 std::vector(可分页内存)做 cudaMemcpyAsync，
    //   驱动会退化为同步暂存拷贝，并且每帧还重新分配/清零输出 vector。
    if (m_trtHostIn) { cudaFreeHost(m_trtHostIn); m_trtHostIn = nullptr; }
    if (m_trtHostOut) { cudaFreeHost(m_trtHostOut); m_trtHostOut = nullptr; }
    cudaHostAlloc((void**)&m_trtHostIn, inFloats * sizeof(float), cudaHostAllocDefault);
    cudaHostAlloc((void**)&m_trtHostOut, outFloats * sizeof(float), cudaHostAllocDefault);
    m_trtHostInFloats = inFloats;
    m_trtHostOutFloats = outFloats;

    buffersAllocated = true;
}

void Detector::FreeBuffers()
{
    if (inference_engine == InferenceEngine::TensorRT) {
        // graph 捕获了固定的 buffer/host 地址，释放前必须先销毁，否则下次重建地址变了会出错
        DestroyTRTGraph();
        m_graphDisabled = false;  // 重新分配后允许再次尝试录制
        if (buffers[0]) { cudaFree(buffers[0]); buffers[0] = nullptr; }
        if (buffers[1]) { cudaFree(buffers[1]); buffers[1] = nullptr; }
        if (m_trtHostIn) { cudaFreeHost(m_trtHostIn); m_trtHostIn = nullptr; }
        if (m_trtHostOut) { cudaFreeHost(m_trtHostOut); m_trtHostOut = nullptr; }
        m_trtHostInFloats = 0;
        m_trtHostOutFloats = 0;
    }
    buffersAllocated = false;
}

// =============================================================
// IoU / NMS
// =============================================================
float Detector::IoU(const cv::Rect& a, const cv::Rect& b)
{
    int interX1 = std::max(a.x, b.x);
    int interY1 = std::max(a.y, b.y);
    int interX2 = std::min(a.x + a.width, b.x + b.width);
    int interY2 = std::min(a.y + a.height, b.y + b.height);
    int interW = interX2 - interX1;
    int interH = interY2 - interY1;
    if (interW <= 0 || interH <= 0) return 0.0f;
    float interArea = (float)(interW * interH);
    float unionArea = (float)(a.area() + b.area()) - interArea;
    return interArea / unionArea;
}

void Detector::NMS(std::vector<Detection>& dets, float iouThreshold)
{
    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) { return a.conf > b.conf; });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> kept;
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (!suppressed[j] && dets[i].classId == dets[j].classId &&
                IoU(dets[i].box, dets[j].box) > iouThreshold)
                suppressed[j] = true;
        }
    }
    dets = std::move(kept);
}

// =============================================================
// Inference (分发到 TensorRT / ONNXRuntime)
// =============================================================
std::vector<Detection> Detector::Inference(const cv::Mat& frame)
{
    if (frame.empty()) return {};
    if (!buffersAllocated) {
        if (inference_engine == InferenceEngine::TensorRT)
            AllocateBuffers();
        else
            buffersAllocated = true;
    }

    const auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<Detection> result = (inference_engine == InferenceEngine::TensorRT)
        ? InferenceTensorRT(frame)
        : InferenceONNXRuntime(frame);

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // 简单滑动统计：每隔 N 帧打印一次平均耗时，避免刷屏
    static int s_frameCnt = 0;
    static double s_totalMsSum = 0.0;
    static double s_inferMsSum = 0.0;
    s_frameCnt++;
    s_totalMsSum += totalMs;
    s_inferMsSum += g_lastPureInferMs;
    if (s_frameCnt >= 60) {
        printf("[TIMING] engine=%s  avg total=%.3fms (%.1f FPS)  avg pure-infer=%.3fms\n",
            inference_engine == InferenceEngine::TensorRT ? "TensorRT" : "ONNXRuntime",
            s_totalMsSum / s_frameCnt, 1000.0 / (s_totalMsSum / s_frameCnt),
            s_inferMsSum / s_frameCnt);
        s_frameCnt = 0;
        s_totalMsSum = 0.0;
        s_inferMsSum = 0.0;
    }

    return result;
}

// =============================================================
// BuildLUT: 一次性初始化 1/255 归一化查找表
// =============================================================
void Detector::BuildLUT()
{
    for (int i = 0; i < 256; i++)
        m_lut[i] = static_cast<float>(i) / 255.0f;
}

// =============================================================
// FastPreprocess: 手写 LetterBox 预处理 (替代 OpenCV 全链路)
//   一次遍历完成: 缩放 + BGR→RGB + 归一化 + HWC→CHW
//   直接写入 m_onnxInputData (m_onnxInputTensor 零拷贝包装此缓冲)
// =============================================================
void Detector::FastPreprocess(const cv::Mat& frame)
{
    const int srcW = frame.cols;
    const int srcH = frame.rows;
    const int dstW = inputSize;
    const int dstH = inputSize;

    // ── 1. 计算 LetterBox 参数 ────────────────────────────────
    m_scale = std::min(static_cast<float>(dstW) / srcW,
        static_cast<float>(dstH) / srcH);
    m_newW = static_cast<int>(srcW * m_scale);
    m_newH = static_cast<int>(srcH * m_scale);
    m_padX = (dstW - m_newW) / 2;
    m_padY = (dstH - m_newH) / 2;

    // ── 2. 平面指针 ─────────────────────────────────────────
    const size_t planeSize = (size_t)dstW * dstH;
    float* rPlane = m_onnxInputData.data();
    float* gPlane = m_onnxInputData.data() + planeSize;
    float* bPlane = m_onnxInputData.data() + planeSize * 2;

    // ── 3. 灰度填充 (LetterBox 背景 114/255) ────────────────
    const float gray = m_lut[114];
    if (m_padX > 0 || m_padY > 0 || m_newW < dstW || m_newH < dstH) {
        std::fill(rPlane, rPlane + planeSize, gray);
        std::fill(gPlane, gPlane + planeSize, gray);
        std::fill(bPlane, bPlane + planeSize, gray);
    }

    if (m_newW <= 0 || m_newH <= 0) return;

    // ── 4. 懒重建内容映射表 (源帧尺寸变化时) ────────────────
    if (srcW != m_lastSrcW || srcH != m_lastSrcH) {
        m_lastSrcW = srcW;
        m_lastSrcH = srcH;

        m_contentMapX.resize(m_newW);
        m_contentMapY.resize(m_newH);

        const float invScale = 1.0f / m_scale;
        for (int c = 0; c < m_newW; c++) {
            int sx = static_cast<int>(c * invScale);
            m_contentMapX[c] = (sx < srcW) ? sx : (srcW - 1);
        }
        for (int r = 0; r < m_newH; r++) {
            int sy = static_cast<int>(r * invScale);
            m_contentMapY[r] = (sy < srcH) ? sy : (srcH - 1);
        }
    }

    const int* mapX = m_contentMapX.data();
    const int* mapY = m_contentMapY.data();
    const uint8_t* srcData = frame.data;
    const int srcStep = static_cast<int>(frame.step);
    const int srcCh = frame.channels();  // 3 for BGR

    // ── 5. 内容区域填充: BGR→Planar RGB + LUT归一化 ────────
    //    4像素展开循环 (参考 others/detector.h 优化)
    for (int r = 0; r < m_newH; r++) {
        const int srcY = mapY[r];
        const uint8_t* srcRow = srcData + srcY * srcStep;
        const int dstRowBase = (m_padY + r) * dstW + m_padX;

        int c = 0;
        const int newW4 = m_newW - 3;
        for (; c < newW4; c += 4) {
            const uint8_t* p0 = srcRow + mapX[c] * srcCh;
            const uint8_t* p1 = srcRow + mapX[c + 1] * srcCh;
            const uint8_t* p2 = srcRow + mapX[c + 2] * srcCh;
            const uint8_t* p3 = srcRow + mapX[c + 3] * srcCh;

            const int di = dstRowBase + c;
            // OpenCV Mat 默认 BGR: p[0]=B, p[1]=G, p[2]=R
            rPlane[di] = m_lut[p0[2]];
            gPlane[di] = m_lut[p0[1]];
            bPlane[di] = m_lut[p0[0]];
            rPlane[di + 1] = m_lut[p1[2]];
            gPlane[di + 1] = m_lut[p1[1]];
            bPlane[di + 1] = m_lut[p1[0]];
            rPlane[di + 2] = m_lut[p2[2]];
            gPlane[di + 2] = m_lut[p2[1]];
            bPlane[di + 2] = m_lut[p2[0]];
            rPlane[di + 3] = m_lut[p3[2]];
            gPlane[di + 3] = m_lut[p3[1]];
            bPlane[di + 3] = m_lut[p3[0]];
        }
        // 尾部 1-3 列
        for (; c < m_newW; c++) {
            const uint8_t* p = srcRow + mapX[c] * srcCh;
            const int di = dstRowBase + c;
            rPlane[di] = m_lut[p[2]];
            gPlane[di] = m_lut[p[1]];
            bPlane[di] = m_lut[p[0]];
        }
    }
}

// =============================================================
// 共用后处理: YOLOv10 [1,300,6]
// =============================================================
std::vector<Detection> Detector::PostProcessV10(const float* output, int numBoxes, int stride,
    int originalW, int originalH, float scale, int padX, int padY)
{
    std::vector<Detection> results;
    const float confThreshold = 0.25f;
    const float nmsThreshold = 0.45f;

    for (int i = 0; i < numBoxes; i++)
    {
        const float* ptr = output + i * stride;

        const float score = ptr[4];
        if (!std::isfinite(score) || score < confThreshold) continue;

        float x1 = ptr[0], y1 = ptr[1], x2 = ptr[2], y2 = ptr[3];
        if (!std::isfinite(x1) || !std::isfinite(y1) ||
            !std::isfinite(x2) || !std::isfinite(y2)) continue;

        if (x1 > x2) std::swap(x1, x2);
        if (y1 > y2) std::swap(y1, y2);

        x1 -= padX; y1 -= padY; x2 -= padX; y2 -= padY;
        x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale;

        x1 = std::max(0.f, std::min(x1, (float)(originalW - 1)));
        y1 = std::max(0.f, std::min(y1, (float)(originalH - 1)));
        x2 = std::max(0.f, std::min(x2, (float)(originalW - 1)));
        y2 = std::max(0.f, std::min(y2, (float)(originalH - 1)));

        if (x2 - x1 < 1.f || y2 - y1 < 1.f) continue;

        Detection det;
        det.box = cv::Rect(cv::Point((int)x1, (int)y1),
            cv::Point((int)x2, (int)y2));
        det.conf = score;
        det.classId = (int)ptr[5];
        results.push_back(det);
    }

    NMS(results, nmsThreshold);
    return results;
}

// =============================================================
// 共用后处理: YOLOv11 [1,4+nc,anchors]
// =============================================================
std::vector<Detection> Detector::PostProcessV11(const float* output, int rowDim, int anchors,
    int originalW, int originalH, float scale, int padX, int padY)
{
    std::vector<Detection> results;
    const float confThreshold = 0.25f;
    const float nmsThreshold = 0.45f;
    const int nc = rowDim - 4;

    const float* row_cx = output + 0 * anchors;
    const float* row_cy = output + 1 * anchors;
    const float* row_w = output + 2 * anchors;
    const float* row_h = output + 3 * anchors;

    for (int i = 0; i < anchors; i++)
    {
        float maxScore = -1.f;
        int   bestCls = 0;
        for (int c = 0; c < nc; c++) {
            float s = output[(4 + c) * anchors + i];
            if (s > maxScore) { maxScore = s; bestCls = c; }
        }
        if (!std::isfinite(maxScore) || maxScore < confThreshold) continue;

        float cx = row_cx[i], cy = row_cy[i];
        float bw = row_w[i], bh = row_h[i];
        if (!std::isfinite(cx) || !std::isfinite(cy) ||
            !std::isfinite(bw) || !std::isfinite(bh)) continue;

        float x1 = cx - bw * 0.5f, y1 = cy - bh * 0.5f;
        float x2 = cx + bw * 0.5f, y2 = cy + bh * 0.5f;

        x1 -= padX; y1 -= padY; x2 -= padX; y2 -= padY;
        x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale;

        x1 = std::max(0.f, std::min(x1, (float)(originalW - 1)));
        y1 = std::max(0.f, std::min(y1, (float)(originalH - 1)));
        x2 = std::max(0.f, std::min(x2, (float)(originalW - 1)));
        y2 = std::max(0.f, std::min(y2, (float)(originalH - 1)));

        if (x2 - x1 < 1.f || y2 - y1 < 1.f) continue;

        Detection det;
        det.box = cv::Rect(cv::Point((int)x1, (int)y1),
            cv::Point((int)x2, (int)y2));
        det.conf = maxScore;
        det.classId = bestCls;
        results.push_back(det);
    }

    NMS(results, nmsThreshold);
    return results;
}

// =============================================================
// TensorRT 推理路径
// =============================================================
std::vector<Detection> Detector::InferenceTensorRT(const cv::Mat& frame)
{
    std::vector<Detection> results;

    // ── A. 快速预处理 (手写, 与 ONNX 路径共用 FastPreprocess) ──
    FastPreprocess(frame);

    const int originalW = frame.cols;
    const int originalH = frame.rows;

    // ── B. planar float → 页锁输入缓冲 (CPU→CPU，graph 内做 H2D) ──
    const size_t inFloats = (size_t)3 * inputSize * inputSize;
    const size_t inBytes = inFloats * sizeof(float);
    std::memcpy(m_trtHostIn, m_onnxInputData.data(), inBytes);

    // ── 输出元素数（按 model_way 确定）──────────────────────────
    const int    rowDim = (model_way == 0) ? 6 : (4 + numClasses);
    const int    anchors = (model_way == 0) ? 300 : numAnchors_;
    const size_t outFloats = (size_t)rowDim * anchors;

    // 固定张量地址（graph 录制 / 普通路径都需要，且必须在 capture 前设好）
    context->setTensorAddress("images", buffers[0]);
    context->setTensorAddress("output0", buffers[1]);

    const auto tInfer0 = std::chrono::high_resolution_clock::now();

    // ── C. 优先走 CUDA Graph：把 H2D→enqueue→D2H 三步合成一次提交 ──
    //   小负载下，多次 cudaMemcpyAsync/enqueueV3 的 CPU 端 launch 开销
    //   是 TRT 单帧延迟高于 DML 的主因；graph 把这些 launch 折叠成一次。
    if (!m_graphReady && !m_graphDisabled) {
        CaptureTRTGraph(outFloats);  // 首次：录制+实例化（内部已含一次真实执行）
    }

    cudaError_t err = cudaSuccess;
    if (m_graphReady && m_graphOutFloats == outFloats) {
        // 已录制：仅启动图 + 一次同步
        cudaGraphLaunch(m_graphExec, stream);
        err = cudaStreamSynchronize(stream);
    }
    else {
        // 回退：原始三步序列（graph 不可用时）
        float* gpu_input = (float*)buffers[0];
        cudaMemcpyAsync(gpu_input, m_trtHostIn, inBytes, cudaMemcpyHostToDevice, stream);
        if (!context->enqueueV3(stream)) {
            printf("[DBG] enqueueV3 返回 false！推理失败\n");
            return results;
        }
        cudaMemcpyAsync(m_trtHostOut, buffers[1], outFloats * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        err = cudaStreamSynchronize(stream);
    }

    {
        const auto tInfer1 = std::chrono::high_resolution_clock::now();
        g_lastPureInferMs = std::chrono::duration<double, std::milli>(tInfer1 - tInfer0).count();
    }
    if (err != cudaSuccess) {
        printf("[DBG] cudaStreamSynchronize 失败: %s\n", cudaGetErrorString(err));
        return results;
    }

    float* output = m_trtHostOut;

    // ── D. 后处理（按 model_way 分支）─────────────────────────────
    if (model_way == 0)
    {
        // YOLOv10: 输出 [1, 300, 6] = [x1,y1,x2,y2,score,class]
        return PostProcessV10(output, anchors, rowDim, originalW, originalH, m_scale, m_padX, m_padY);
    }

    // ── 原始输出统计（只在首帧打印，确认数值范围）────────────
    static bool s_firstFrame = true;
    if (s_firstFrame) {
        s_firstFrame = false;
        float minV = output[0], maxV = output[0], sumV = 0.f;
        int   nanCnt = 0;
        for (size_t i = 0; i < outFloats; i++) {
            float v = output[i];
            if (!std::isfinite(v)) { nanCnt++; continue; }
            minV = std::min(minV, v); maxV = std::max(maxV, v); sumV += v;
        }
        printf("[DBG] output raw  min=%.4f  max=%.4f  mean=%.4f  nan/inf=%d  total=%zu\n",
            minV, maxV, sumV / (float)outFloats, nanCnt, outFloats);

        float maxConf = -1.f;
        for (int c = 0; c < numClasses; c++) {
            const float* row = output + (4 + c) * numAnchors_;
            for (int i = 0; i < numAnchors_; i++)
                maxConf = std::max(maxConf, row[i]);
        }
        printf("[DBG] max conf across all anchors/classes = %.4f  (threshold=0.25)\n", maxConf);

        float maxCx = -1.f;
        const float* rowCx = output + 0 * numAnchors_;
        for (int i = 0; i < numAnchors_; i++)
            maxCx = std::max(maxCx, rowCx[i]);
        printf("[DBG] max cx = %.2f  (should be ~0..%d)\n", maxCx, inputSize);
    }

    return PostProcessV11(output, rowDim, numAnchors_, originalW, originalH, m_scale, m_padX, m_padY);
}

// =============================================================
// CaptureTRTGraph: 录制 H2D→enqueueV3→D2H 为 CUDA Graph
// =============================================================
void Detector::CaptureTRTGraph(size_t outFloats)
{
    float* gpu_input = (float*)buffers[0];
    const size_t inBytes = (size_t)3 * inputSize * inputSize * sizeof(float);

    // ── 1. 先做一次"热身"真实执行：完成 TRT 可能的惰性初始化/选 kernel，
    //       否则首次 capture 可能把一次性初始化也录进图里或导致 capture 失败。
    cudaMemcpyAsync(gpu_input, m_trtHostIn, inBytes, cudaMemcpyHostToDevice, stream);
    if (!context->enqueueV3(stream)) {
        printf("[Graph] 热身 enqueueV3 失败，禁用 graph，回退普通路径\n");
        m_graphDisabled = true;
        return;
    }
    cudaMemcpyAsync(m_trtHostOut, buffers[1], outFloats * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        printf("[Graph] 热身同步失败，禁用 graph\n");
        m_graphDisabled = true;
        return;
    }

    // ── 2. 录制 ─────────────────────────────────────────────────
    cudaError_t e = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        printf("[Graph] BeginCapture 失败: %s，禁用 graph\n", cudaGetErrorString(e));
        m_graphDisabled = true;
        return;
    }
    cudaMemcpyAsync(gpu_input, m_trtHostIn, inBytes, cudaMemcpyHostToDevice, stream);
    bool enqOk = context->enqueueV3(stream);
    cudaMemcpyAsync(m_trtHostOut, buffers[1], outFloats * sizeof(float),
        cudaMemcpyDeviceToHost, stream);

    cudaGraph_t graph = nullptr;
    cudaError_t endE = cudaStreamEndCapture(stream, &graph);

    if (!enqOk || endE != cudaSuccess || graph == nullptr) {
        printf("[Graph] 录制失败(enq=%d, end=%s)，禁用 graph，回退普通路径\n",
            (int)enqOk, cudaGetErrorString(endE));
        if (graph) cudaGraphDestroy(graph);
        m_graphDisabled = true;
        return;
    }

    // ── 3. 实例化 ───────────────────────────────────────────────
    cudaGraphExec_t exec = nullptr;
    cudaError_t instE = cudaGraphInstantiate(&exec, graph, 0);
    if (instE != cudaSuccess || exec == nullptr) {
        printf("[Graph] Instantiate 失败: %s，禁用 graph\n", cudaGetErrorString(instE));
        cudaGraphDestroy(graph);
        m_graphDisabled = true;
        return;
    }

    m_graph = graph;
    m_graphExec = exec;
    m_graphOutFloats = outFloats;
    m_graphReady = true;
    printf("[Graph] CUDA Graph 录制成功，后续帧走 graph launch（outFloats=%zu）\n", outFloats);
}

void Detector::DestroyTRTGraph()
{
    if (m_graphExec) { cudaGraphExecDestroy(m_graphExec); m_graphExec = nullptr; }
    if (m_graph) { cudaGraphDestroy(m_graph); m_graph = nullptr; }
    m_graphReady = false;
    m_graphOutFloats = 0;
}

// =============================================================
// ONNXRuntime 推理路径 (快速版: 预分配缓冲 + 手写预处理 + 零拷贝Tensor复用)
// =============================================================
std::vector<Detection> Detector::InferenceONNXRuntime(const cv::Mat& frame)
{
    std::vector<Detection> results;

    // ── A. 快速预处理 (手写, 直接写入 m_onnxInputData) ──────
    FastPreprocess(frame);

    const int originalW = frame.cols;
    const int originalH = frame.rows;

    // ── B. 推理 (复用预创建的 tensor 和 RunOptions) ────────
    std::vector<Ort::Value> outputTensors;
    const auto tInfer0 = std::chrono::high_resolution_clock::now();
    try {
        outputTensors = ortSession.Run(
            m_onnxRunOptions,
            &m_pInputName, &m_onnxInputTensor, 1,
            &m_pOutputName, 1);
    }
    catch (const Ort::Exception& e) {
        printf("[DBG] ONNXRuntime Run 失败: %s\n", e.what());
        return results;
    }
    const auto tInfer1 = std::chrono::high_resolution_clock::now();
    g_lastPureInferMs = std::chrono::duration<double, std::milli>(tInfer1 - tInfer0).count();

    if (outputTensors.empty()) return results;

    const float* output = outputTensors[0].GetTensorData<float>();
    auto outShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();

    if (outShape.size() != 3) return results;
    int d1 = (int)outShape[1];
    int d2 = (int)outShape[2];

    // ── C. 后处理 (使用 FastPreprocess 计算好的 letterbox 参数) ──
    if (model_way == 0)
    {
        // YOLOv10: [1, num_boxes, 6]
        return PostProcessV10(output, d1, d2, originalW, originalH, m_scale, m_padX, m_padY);
    }
    else
    {
        // YOLOv11: [1, 4+nc, anchors]
        return PostProcessV11(output, d1, d2, originalW, originalH, m_scale, m_padX, m_padY);
    }
}