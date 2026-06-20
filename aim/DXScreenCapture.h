#ifndef DXSCREEN_CAPTURE_H
#define DXSCREEN_CAPTURE_H

#include "IScreenCapture.h"
#include <immintrin.h>
#include <atomic>

class DXScreenCapture {
public:
    DXScreenCapture();
    ~DXScreenCapture();

    // 公开接口
    bool Initialize(int width, int height);
    bool InitializeRegion(int x, int y, int width, int height);
    bool SetRegion(int x, int y, int width, int height);

    std::vector<uint8_t> CaptureBGR();
    bool CaptureBGR(uint8_t* outBuffer, size_t bufferSize);
    std::vector<uint8_t> CaptureBMP();

    void Release();

    // 获取信息
    int GetWidth() const { return outWidth; }
    int GetHeight() const { return outHeight; }
    int GetChannels() const { return outChannels; }

private:
    // DXGI相关
    bool InitializeDXGI();
    void CleanupDXGI();
    bool InitializeTextures();
    void CleanupTextureCache();
    bool UpdateFrame(UINT timeoutMs = INFINITE);  // DXGI 阻塞等下一帧（默认 INFINITE = 完全跟随游戏，CPU≈0）
    void ReleaseFrame();

    // 错误恢复
    bool OnOutputChange();
    bool ReinitializeDuplication();
    void RecalculateCenterPosition();

    // 纹理管理
    CachedTexturePair* GetCachedTexturePair(UINT width, UINT height);

    // 格式转换 - 改为直接填充到 vector
    bool ConvertToBMP_Fast(const BYTE* srcData, int width, int height, int srcPitch, std::vector<uint8_t>& outBuffer);
    void ConvertBGRA2BGR_SIMD(const uint8_t* bgra, uint8_t* bgr, int width, int height, int stride);

    // 辅助函数
    bool IsSSE42Supported();
    inline bool ValidateBMPParameters(const BYTE* srcData, int width, int height, int srcPitch);
    inline bool IsResolutionFirstCall(int width, int height);

    // DXGI对象
    ID3D11Device* pd3dDevice;
    ID3D11DeviceContext* pd3dContext;
    IDXGIOutputDuplication* pDeskDupl;
    IDXGIOutput* pDXGIOutput;
    IDXGIAdapter1* pDXGIAdapter;
    IDXGIFactory1* pDXGIFactory;
    ID3D11Texture2D* frameTexture;

    // 捕获区域
    int captureX;
    int captureY;
    int captureWidth;
    int captureHeight;
    int outWidth;
    int outHeight;
    int outChannels;

    // 纹理缓存
    CachedTexturePair texturePair_general;
    CachedTexturePair* currentTexturePair;

    // 状态标志
    bool textureInitialized;
    bool initialized;
    bool updated;
    bool sse42Supported;
    bool useCenterMode;

    // 屏幕信息
    int screenWidth;
    int screenHeight;

    // 缓存
    std::unordered_map<std::pair<int, int>, BMPHeaderCache, PairHash> bmpHeaderCache;
    std::vector<ResolutionFirstCall> resolutionFirstCalls;

    // 统计
    int frameCounter;
    std::atomic<int> reinitCount;
};

#endif // DXSCREEN_CAPTURE_H
