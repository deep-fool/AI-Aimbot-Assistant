#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include "onnxruntime_cxx_api.h"  // 本地 1.21.0 版本 (匹配 DirectML DLL)

struct Detection {
    cv::Rect box;
    float conf;
    int classId;
};

struct match {
    int centerX;
    int centerY;
    float vx;
    float vy;
    float w;
    float h;
};

// 推理引擎类型：
//   0 = TensorRT  (.engine 文件，model_way 决定 v10/v11 输出格式)
//   1 = ONNXRuntime (.onnx 文件，model_way 决定 v10/v11 输出格式)
enum class InferenceEngine {
    TensorRT = 0,
    ONNXRuntime = 1
};

class Detector {
public:
    // engineType: TensorRT 或 ONNXRuntime
    // modelWay  : 0 = YOLOv10 (输出 [1,300,6] 绝对像素 xyxy), 1 = YOLOv11 (输出 [1,4+nc,anchors])
    Detector(const std::string& modelPath, int modelWay = 1,
        InferenceEngine engineType = InferenceEngine::TensorRT);
    ~Detector();

    void SetInputSize(int size);
    void SetONNXDevice(int device) { m_onnxDevicePref = device; }  // 0=Auto 1=DML 2=CUDA 3=CPU
    std::vector<Detection> Inference(const cv::Mat& frame);

private:
    // ---- TensorRT ----
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    void* buffers[2];
    cudaStream_t stream;

    // ---- TensorRT 固定(页锁)主机缓冲：避免 pageable cudaMemcpyAsync 退化为同步慢拷贝 ----
    float* m_trtHostIn = nullptr;     // 页锁输入缓冲 (3*IS*IS)，单次连续 H2D
    float* m_trtHostOut = nullptr;    // 页锁输出缓冲，复用避免每帧 vector 分配
    size_t m_trtHostInFloats = 0;
    size_t m_trtHostOutFloats = 0;

    // ---- CUDA Graph：把 H2D→enqueue→D2H 录制成图，消除每帧多次 API 的 CPU launch 开销 ----
    //   适用前提：输入 shape 固定、输入/输出地址固定(页锁缓冲 + 固定 device buffer)。
    cudaGraph_t      m_graph = nullptr;
    cudaGraphExec_t  m_graphExec = nullptr;
    bool             m_graphReady = false;     // 已成功录制并实例化
    bool             m_graphDisabled = false;  // 录制失败则回退普通路径，不再重试
    size_t           m_graphOutFloats = 0;     // 录制时的输出元素数(用于一致性校验)

    // ---- ONNXRuntime ----
    Ort::Env ortEnv{ nullptr };
    Ort::Session ortSession{ nullptr };
    Ort::MemoryInfo ortMemInfo{ nullptr };
    std::string ortInputName;
    std::string ortOutputName;

    // ---- ONNXRuntime 快速路径 (预分配 + 零拷贝 + 手写预处理) ----
    std::vector<float> m_onnxInputData;            // 预分配 planar float 缓冲 (3*IS*IS)
    Ort::Value m_onnxInputTensor{ nullptr };        // 预创建 tensor, 包装 m_onnxInputData (零拷贝)
    Ort::RunOptions m_onnxRunOptions;               // 预创建 RunOptions, 每帧复用
    const char* m_pInputName = nullptr;             // 缓存 c_str 指针
    const char* m_pOutputName = nullptr;
    std::array<int64_t, 4> m_onnxInputShape{};      // 预构建 shape {1,3,IS,IS}

    // 预处理查找表
    float m_lut[256];                               // i/255.0f 归一化表
    std::vector<int> m_contentMapX;                 // 内容区列映射: srcX = map[c]
    std::vector<int> m_contentMapY;                 // 内容区行映射: srcY = map[r]
    int m_lastSrcW = 0;                             // 跟踪源帧宽度变化, 懒重建映射表
    int m_lastSrcH = 0;

    // Letterbox 参数 (FastPreprocess 计算, PostProcess 消费)
    float m_scale = 1.0f;
    int m_padX = 0, m_padY = 0;
    int m_newW = 0, m_newH = 0;

    void BuildLUT();                                // 一次性初始化 m_lut
    void FastPreprocess(const cv::Mat& frame);      // 手写预处理 → m_onnxInputData

    int inputSize = 320;
    int numClasses = 1;
    int numAnchors_ = 8400;   // ← 从 engine/onnx 输出形状自动推导
    int model_way = 1;        // 0=YOLOv10  1=YOLOv11
    InferenceEngine inference_engine = InferenceEngine::TensorRT; // 0=TensorRT 1=ONNXRuntime
    int m_onnxDevicePref = 0;    // 0=Auto(DML→CUDA→CPU) 1=DirectML 2=CUDA 3=CPU

    bool buffersAllocated = false;

    void AllocateBuffers();
    void FreeBuffers();

    // 录制 H2D→enqueue→D2H 为 CUDA Graph；失败则设置 m_graphDisabled 回退普通路径。
    void CaptureTRTGraph(size_t outFloats);
    void DestroyTRTGraph();

    void InitTensorRT(const std::string& modelPath);
    void InitONNXRuntime(const std::string& modelPath);

    std::vector<Detection> InferenceTensorRT(const cv::Mat& frame);
    std::vector<Detection> InferenceONNXRuntime(const cv::Mat& frame);

    // 共用后处理：letterbox 参数 + 原始模型输出 → Detection 列表
    std::vector<Detection> PostProcessV10(const float* output, int numBoxes, int stride,
        int originalW, int originalH, float scale, int padX, int padY);
    std::vector<Detection> PostProcessV11(const float* output, int rowDim, int anchors,
        int originalW, int originalH, float scale, int padX, int padY);

    static float IoU(const cv::Rect& a, const cv::Rect& b);
    static void  NMS(std::vector<Detection>& dets, float iouThreshold);
};

class TRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING)
            printf("[TRT] %s\n", msg);
    }
} static gLogger;