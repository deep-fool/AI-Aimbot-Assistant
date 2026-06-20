#pragma once
#include <NvInfer.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <cuda_runtime_api.h>
#include <onnxruntime_cxx_api.h>

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
    std::vector<Detection> Inference(const cv::Mat& frame);

private:
    // ---- TensorRT ----
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;

    void* buffers[2];
    cudaStream_t stream;

    // ---- ONNXRuntime ----
    Ort::Env ortEnv{ nullptr };
    Ort::Session ortSession{ nullptr };
    Ort::MemoryInfo ortMemInfo{ nullptr };
    std::string ortInputName;
    std::string ortOutputName;

    int inputSize = 320;
    int numClasses = 1;
    int numAnchors_ = 8400;   // ← 从 engine/onnx 输出形状自动推导
    int model_way = 1;        // 0=YOLOv10  1=YOLOv11
    InferenceEngine inference_engine = InferenceEngine::TensorRT; // 0=TensorRT 1=ONNXRuntime

    bool buffersAllocated = false;

    void AllocateBuffers();
    void FreeBuffers();

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