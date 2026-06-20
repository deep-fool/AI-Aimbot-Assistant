#ifndef WGC_SCREEN_CAPTURE_H
#define WGC_SCREEN_CAPTURE_H

#include <Windows.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Metadata.h>
#include <windows.graphics.capture.h>
#include <windows.graphics.capture.interop.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <memory>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <immintrin.h>
#include "IScreenCapture.h"

#pragma comment(lib, "windowsapp.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

namespace winrt {
    namespace Windows::Graphics::Capture {
        struct GraphicsCaptureItem;
        struct Direct3D11CaptureFramePool;
        struct GraphicsCaptureSession;
        struct Direct3D11CaptureFrame;
    }
    namespace Windows::Graphics::DirectX {
        enum class DirectXPixelFormat;
    }
    namespace Windows::Graphics::DirectX::Direct3D11 {
        struct IDirect3DDevice;
    }
}

std::wstring CharToWString(const char* str);
extern "C" {
    HRESULT __stdcall CreateDirect3D11DeviceFromDXGIDevice(::IDXGIDevice* dxgiDevice, ::IInspectable** graphicsDevice);
}

template <typename T>
winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object);

class WGCScreenCapture {
public:
    WGCScreenCapture();
    ~WGCScreenCapture();

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
    int GetWidth() const { return outWidth; }
    int GetHeight() const { return outHeight; }
    int GetChannels() const { return outChannels; }

private:
    using Direct3DDevice = winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
    using CaptureItem = winrt::Windows::Graphics::Capture::GraphicsCaptureItem;
    using FramePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool;
    using CaptureSession = winrt::Windows::Graphics::Capture::GraphicsCaptureSession;
    using CaptureFrame = winrt::Windows::Graphics::Capture::Direct3D11CaptureFrame;
    using DirectXPixelFormat = winrt::Windows::Graphics::DirectX::DirectXPixelFormat;

    bool InitializeWGC();
    void CleanupWGC();
    bool InitializeTextures();
    void CleanupTextureCache();
    CachedTexturePair* GetCachedTexturePair(UINT width, UINT height);

    bool UpdateFrame();
    void ReleaseFrame();
    void RecalculateCenterPosition();

    bool ConvertToBMP_Fast(const BYTE* srcData, int width, int height, int srcPitch, std::vector<uint8_t>& outBuffer);
    void ConvertBGRA2BGR_SIMD(const uint8_t* bgra, uint8_t* bgr, int width, int height, int stride);

    bool IsSSE42Supported();
    inline bool ValidateBMPParameters(const BYTE* srcData, int width, int height, int srcPitch);
    inline bool IsResolutionFirstCall(int width, int height);

    winrt::com_ptr<ID3D11Device> CreateD3DDevice();
    Direct3DDevice CreateDirect3DDevice();
    CaptureItem CreateCaptureItemForWindow(HWND hwnd);
    CaptureItem CreateCaptureItemForMonitor(HMONITOR hmonitor);
    bool StartCapture();
    void StopCapture();
    bool ResizeIfNeeded();

    ID3D11Device* pd3dDevice;
    ID3D11DeviceContext* pd3dContext;

    CaptureItem captureItem;
    FramePool framePool;
    CaptureSession captureSession;
    Direct3DDevice device;
    DirectXPixelFormat pixelFormat;
    winrt::Windows::Graphics::SizeInt32 lastSize;

    ID3D11Texture2D* frameTexture;

    int captureX;
    int captureY;
    int captureWidth;
    int captureHeight;

    int outWidth;
    int outHeight;
    int outChannels;

    CachedTexturePair texturePair_general;
    CachedTexturePair* currentTexturePair;
    bool textureInitialized;

    bool initialized;
    bool updated;
    std::atomic<bool> isRunning;
    std::atomic<bool> closed;

    bool sse42Supported;

    int screenWidth;
    int screenHeight;
    bool monitorMode;
    HWND targetWindow;
    HMONITOR hmonitor;

    std::unordered_map<std::pair<int, int>, BMPHeaderCache, PairHash> bmpHeaderCache;
    std::vector<ResolutionFirstCall> resolutionFirstCalls;

    int frameCounter;
    bool useCenterMode;
};

#endif
