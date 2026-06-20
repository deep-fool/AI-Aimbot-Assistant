#include "Detector.h"
#include <fstream>
#include <stdexcept>
#include <numeric>
#include <algorithm>
#include <chrono>

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

    // 尝试启用 CUDA EP，失败则回退到 CPU
    try {
        OrtCUDAProviderOptions cudaOpts{};
        sessionOptions.AppendExecutionProvider_CUDA(cudaOpts);
        printf("[DBG] ONNXRuntime: CUDA EP 已启用\n");
    }
    catch (const std::exception& e) {
        printf("[DBG] ONNXRuntime: CUDA EP 启用失败 (%s)，使用 CPU\n", e.what());
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
    if (inference_engine == InferenceEngine::TensorRT)
        AllocateBuffers();
    else
        buffersAllocated = true; // ONNXRuntime 不需要预分配 GPU buffer
}

// =============================================================
// AllocateBuffers / FreeBuffers (仅 TensorRT 使用)
// =============================================================
void Detector::AllocateBuffers()
{
    cudaMalloc(&buffers[0],
        static_cast<size_t>(1) * 3 * inputSize * inputSize * sizeof(float));

    if (model_way == 0) {
        // YOLOv10: 输出 [1, 300, 6]，固定 300 个候选框
        cudaMalloc(&buffers[1],
            static_cast<size_t>(1) * 300 * 6 * sizeof(float));
        printf("[DBG] AllocateBuffers(v10): inputSize=%d  boxes=300 stride=6\n", inputSize);
    }
    else {
        // YOLOv11: 按实际 anchor 数分配
        const int rowDim = 4 + numClasses;
        cudaMalloc(&buffers[1],
            static_cast<size_t>(1) * rowDim * numAnchors_ * sizeof(float));
        printf("[DBG] AllocateBuffers(v11): inputSize=%d  rowDim=%d  numAnchors=%d\n",
            inputSize, rowDim, numAnchors_);
    }
    buffersAllocated = true;
}

void Detector::FreeBuffers()
{
    if (inference_engine == InferenceEngine::TensorRT) {
        if (buffers[0]) { cudaFree(buffers[0]); buffers[0] = nullptr; }
        if (buffers[1]) { cudaFree(buffers[1]); buffers[1] = nullptr; }
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

    // ── A. LetterBox ─────────────────────────────────────────
    const int originalW = frame.cols;
    const int originalH = frame.rows;
    const float scale = std::min(
        (float)inputSize / originalW,
        (float)inputSize / originalH);
    const int newW = (int)(originalW * scale);
    const int newH = (int)(originalH * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    cv::Mat input(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    const int padX = (inputSize - newW) / 2;
    const int padY = (inputSize - newH) / 2;
    resized.copyTo(input(cv::Rect(padX, padY, newW, newH)));
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

    // ── B. 归一化 ─────────────────────────────────────────────
    cv::Mat inputF;
    input.convertTo(inputF, CV_32FC3, 1.0 / 255.0);

    // ── C. HWC -> CHW ─────────────────────────────────────────
    std::vector<cv::Mat> channels(3);
    cv::split(inputF, channels);
    float* gpu_input = (float*)buffers[0];
    const size_t channelBytes = (size_t)inputSize * inputSize * sizeof(float);
    for (int i = 0; i < 3; i++)
        cudaMemcpyAsync(gpu_input + i * inputSize * inputSize,
            channels[i].data, channelBytes,
            cudaMemcpyHostToDevice, stream);

    // ── D. 推理 ───────────────────────────────────────────────
    context->setTensorAddress("images", buffers[0]);
    context->setTensorAddress("output0", buffers[1]);

    const auto tInfer0 = std::chrono::high_resolution_clock::now();
    bool ok = context->enqueueV3(stream);
    if (!ok) {
        printf("[DBG] enqueueV3 返回 false！推理失败\n");
        return results;
    }

    // ── E. GPU -> CPU & 后处理（按 model_way 分支） ─────────────
    if (model_way == 0)
    {
        // =========================================================
        // YOLOv10: 输出 [1, 300, 6] = [x1,y1,x2,y2,score,class]
        // 坐标为 letterbox 空间内的绝对像素坐标
        // =========================================================
        const int numBoxes = 300;
        const int stride = 6;
        std::vector<float> output((size_t)numBoxes * stride);

        cudaMemcpyAsync(output.data(), buffers[1],
            output.size() * sizeof(float),
            cudaMemcpyDeviceToHost, stream);
        cudaError_t err = cudaStreamSynchronize(stream);
        {
            const auto tInfer1 = std::chrono::high_resolution_clock::now();
            g_lastPureInferMs = std::chrono::duration<double, std::milli>(tInfer1 - tInfer0).count();
        }
        if (err != cudaSuccess) {
            printf("[DBG] cudaStreamSynchronize 失败: %s\n", cudaGetErrorString(err));
            return results;
        }

        return PostProcessV10(output.data(), numBoxes, stride, originalW, originalH, scale, padX, padY);
    }

    // =========================================================
    // YOLOv11: 输出 [1, 4+nc, anchors]
    // =========================================================
    const int rowDim = 4 + numClasses;
    std::vector<float> output((size_t)rowDim * numAnchors_);
    cudaMemcpyAsync(output.data(), buffers[1],
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost, stream);
    cudaError_t err = cudaStreamSynchronize(stream);
    {
        const auto tInfer1 = std::chrono::high_resolution_clock::now();
        g_lastPureInferMs = std::chrono::duration<double, std::milli>(tInfer1 - tInfer0).count();
    }
    if (err != cudaSuccess) {
        printf("[DBG] cudaStreamSynchronize 失败: %s\n", cudaGetErrorString(err));
        return results;
    }

    // ── F. 原始输出统计（只在首帧打印，确认数值范围）────────────
    static bool s_firstFrame = true;
    if (s_firstFrame) {
        s_firstFrame = false;
        float minV = output[0], maxV = output[0], sumV = 0.f;
        int   nanCnt = 0;
        for (float v : output) {
            if (!std::isfinite(v)) { nanCnt++; continue; }
            minV = std::min(minV, v); maxV = std::max(maxV, v); sumV += v;
        }
        printf("[DBG] output raw  min=%.4f  max=%.4f  mean=%.4f  nan/inf=%d  total=%zu\n",
            minV, maxV, sumV / output.size(), nanCnt, output.size());

        float maxConf = -1.f;
        for (int c = 0; c < numClasses; c++) {
            const float* row = output.data() + (4 + c) * numAnchors_;
            for (int i = 0; i < numAnchors_; i++)
                maxConf = std::max(maxConf, row[i]);
        }
        printf("[DBG] max conf across all anchors/classes = %.4f  (threshold=0.25)\n", maxConf);

        float maxCx = -1.f;
        const float* rowCx = output.data() + 0 * numAnchors_;
        for (int i = 0; i < numAnchors_; i++)
            maxCx = std::max(maxCx, rowCx[i]);
        printf("[DBG] max cx = %.2f  (should be ~0..%d)\n", maxCx, inputSize);
    }

    return PostProcessV11(output.data(), rowDim, numAnchors_, originalW, originalH, scale, padX, padY);
}

// =============================================================
// ONNXRuntime 推理路径
// =============================================================
std::vector<Detection> Detector::InferenceONNXRuntime(const cv::Mat& frame)
{
    std::vector<Detection> results;

    // ── A. LetterBox ─────────────────────────────────────────
    const int originalW = frame.cols;
    const int originalH = frame.rows;
    const float scale = std::min(
        (float)inputSize / originalW,
        (float)inputSize / originalH);
    const int newW = (int)(originalW * scale);
    const int newH = (int)(originalH * scale);

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newW, newH));

    cv::Mat input(inputSize, inputSize, CV_8UC3, cv::Scalar(114, 114, 114));
    const int padX = (inputSize - newW) / 2;
    const int padY = (inputSize - newH) / 2;
    resized.copyTo(input(cv::Rect(padX, padY, newW, newH)));
    cv::cvtColor(input, input, cv::COLOR_BGR2RGB);

    // ── B. 归一化 ─────────────────────────────────────────────
    cv::Mat inputF;
    input.convertTo(inputF, CV_32FC3, 1.0 / 255.0);

    // ── C. HWC -> CHW ─────────────────────────────────────────
    std::vector<cv::Mat> channels(3);
    cv::split(inputF, channels);

    std::vector<float> inputTensorValues((size_t)3 * inputSize * inputSize);
    const size_t planeSize = (size_t)inputSize * inputSize;
    for (int c = 0; c < 3; c++)
        std::memcpy(inputTensorValues.data() + c * planeSize,
            channels[c].data, planeSize * sizeof(float));

    // ── D. 构造输入 Tensor 并推理 ────────────────────────────
    std::array<int64_t, 4> inputShape = { 1, 3, inputSize, inputSize };

    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        ortMemInfo, inputTensorValues.data(), inputTensorValues.size(),
        inputShape.data(), inputShape.size());

    const char* inputNames[] = { ortInputName.c_str() };
    const char* outputNames[] = { ortOutputName.c_str() };

    std::vector<Ort::Value> outputTensors;
    const auto tInfer0 = std::chrono::high_resolution_clock::now();
    try {
        outputTensors = ortSession.Run(Ort::RunOptions{ nullptr },
            inputNames, &inputTensor, 1,
            outputNames, 1);
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
    // outShape: [1, d1, d2]

    if (outShape.size() != 3) return results;
    int d1 = (int)outShape[1];
    int d2 = (int)outShape[2];

    // ── E. 后处理（按 model_way 分支）────────────────────────
    if (model_way == 0)
    {
        // YOLOv10: [1, num_boxes, 6]
        int numBoxes = d1;
        int stride = d2; // 应为 6
        return PostProcessV10(output, numBoxes, stride, originalW, originalH, scale, padX, padY);
    }
    else
    {
        // YOLOv11: [1, 4+nc, anchors]
        int rowDim = d1;
        int anchors = d2;
        return PostProcessV11(output, rowDim, anchors, originalW, originalH, scale, padX, padY);
    }
}