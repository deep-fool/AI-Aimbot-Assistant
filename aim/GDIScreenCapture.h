#ifndef GDI_SCREEN_CAPTURE_H
#define GDI_SCREEN_CAPTURE_H

#define NOMINMAX
#include <memory>
#include <vector>
#include <cstdint>
#include <atomic>
#include <unordered_map>
#include "IScreenCapture.h"
#include <Windows.h>
#include <immintrin.h>

struct GDIResource {
    HDC screenDC = nullptr;
    HDC windowDC = nullptr;
    HDC memoryDC = nullptr;
    HBITMAP bitmap = nullptr;
    HWND targetWindow = nullptr;

    GDIResource() = default;
    ~GDIResource();

    bool Initialize(HWND target, int width, int height);
    void Release();

    GDIResource(const GDIResource&) = delete;
    GDIResource& operator=(const GDIResource&) = delete;
};

class GDIScreenCapture {
public:
    GDIScreenCapture();
    ~GDIScreenCapture();

    // 初始化接口
    bool Initialize(int width, int height);
    bool InitializeRegion(int x, int y, int width, int height);
    bool SetWindow(HWND hwnd);
    bool SetRegion(int x, int y, int width, int height);
    void Release();

    std::vector<uint8_t> CaptureBGR();
    bool CaptureBGR(uint8_t* outBuffer, size_t bufferSize);
    std::vector<uint8_t> CaptureBMP();

    // 获取信息
    int GetWidth() const { return captureWidth; }
    int GetHeight() const { return captureHeight; }

private:
    bool IsSSE42Supported();
    inline bool ValidateBMPParameters(const BYTE* srcData, int x, int y, int width, int height, int srcPitch);
    inline bool IsResolutionFirstCall(int width, int height);

    bool PerformCapture();
    bool ExtractBGRData(std::vector<uint8_t>& outBuffer);
    bool ExtractBMPData(std::vector<uint8_t>& outBuffer);
    bool UpdateBitmapIfNeeded(int width, int height);

    void CheckAndUpdateScreenResolution();
    void RecalculateCenterPosition();

    void ConvertDIBtoBGR_SIMD(const uint8_t* src, uint8_t* dst, int width, int height, int srcStride);

    GDIResource gdiResources;

    BITMAPINFOHEADER bitmapInfo;

    int captureX{ 0 };
    int captureY{ 0 };
    int captureWidth{ 0 };
    int captureHeight{ 0 };
    int captureTimeoutMs{ 0 };
    bool useCustomRegion{ false };
    bool useCenterMode{ false };

    int lastScreenWidth{ 0 };
    int lastScreenHeight{ 0 };
    int originalWidth{ 0 };
    int originalHeight{ 0 };

    std::vector<uint8_t> dibBuffer;

    std::atomic<bool> initialized{ false };
    bool sse42Supported;

    std::unordered_map<std::pair<int, int>, BMPHeaderCache, PairHash> bmpHeaderCache;
    std::vector<ResolutionFirstCall> resolutionFirstCalls;
};

#endif
