#define NOMINMAX
#include "WGCScreenCapture.h"
#include <algorithm>
#include <intrin.h>
#include <thread>

// WinRT 命名空间
namespace winrt {
    using namespace Windows::Foundation;
    using namespace Windows::Foundation::Metadata;
    using namespace Windows::Graphics::Capture;
    using namespace Windows::Graphics::DirectX;
    using namespace Windows::Graphics::DirectX::Direct3D11;
    using namespace Windows::System;
}

// IDirect3DDxgiInterfaceAccess 接口定义
struct __declspec(uuid("A9B3D012-3DF2-4EE3-B8D1-8695F457D3C1"))
    IDirect3DDxgiInterfaceAccess : ::IUnknown {
    virtual HRESULT __stdcall GetInterface(GUID const& id, void** object) = 0;
};

//============================================================================
// 辅助函数实现
//============================================================================

std::wstring CharToWString(const char* str) {
    if (!str) return L"";
    int size_needed = MultiByteToWideChar(CP_ACP, 0, str, -1, NULL, 0);
    std::wstring wstr(size_needed, 0);
    MultiByteToWideChar(CP_ACP, 0, str, -1, &wstr[0], size_needed);
    return wstr;
}

template <typename T>
winrt::com_ptr<T> GetDXGIInterfaceFromObject(winrt::Windows::Foundation::IInspectable const& object) {
    auto access = object.as<IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<T> result;
    winrt::check_hresult(access->GetInterface(winrt::guid_of<T>(), result.put_void()));
    return result;
}

//============================================================================
// WGCScreenCapture 类实现
//============================================================================

WGCScreenCapture::WGCScreenCapture()
    : pd3dDevice(nullptr)
    , pd3dContext(nullptr)
    , captureItem(nullptr)
    , framePool(nullptr)
    , captureSession(nullptr)
    , device(nullptr)
    , frameTexture(nullptr)
    , captureX(0)
    , captureY(0)
    , captureWidth(0)
    , captureHeight(0)
    , outWidth(0)
    , outHeight(0)
    , outChannels(3)
    , currentTexturePair(nullptr)
    , textureInitialized(false)
    , initialized(false)
    , updated(false)
    , isRunning(false)
    , closed(false)
    , sse42Supported(false)
    , screenWidth(0)
    , screenHeight(0)
    , monitorMode(true)
    , targetWindow(nullptr)
    , hmonitor(nullptr)
    , frameCounter(0)
    , useCenterMode(false) {

    sse42Supported = IsSSE42Supported();
    pixelFormat = winrt::Windows::Graphics::DirectX::DirectXPixelFormat::B8G8R8A8UIntNormalized;
}

WGCScreenCapture::~WGCScreenCapture() {
    Release();
}

bool WGCScreenCapture::IsSSE42Supported() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] & (1 << 20)) != 0;
}

inline bool WGCScreenCapture::ValidateBMPParameters(const BYTE* srcData, int width, int height, int srcPitch) {
    return srcData && width > 0 && height > 0 && srcPitch > 0;
}

inline bool WGCScreenCapture::IsResolutionFirstCall(int width, int height) {
    for (auto& rc : resolutionFirstCalls) {
        if (rc.IsFirstCall(width, height)) {
            return true;
        }
    }
    return false;
}

bool WGCScreenCapture::InitializeWGC() {
    if (initialized) return true;

    try {
        winrt::com_ptr<ID3D11Device> tempDevice = CreateD3DDevice();
        if (!tempDevice) return false;

        pd3dDevice = tempDevice.get();
        pd3dDevice->AddRef();
        pd3dDevice->GetImmediateContext(&pd3dContext);

        device = CreateDirect3DDevice();
        if (!device) return false;

        HMONITOR hMonitor = MonitorFromPoint({ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
        MONITORINFO monitorInfo = {};
        monitorInfo.cbSize = sizeof(MONITORINFO);
        if (GetMonitorInfo(hMonitor, &monitorInfo)) {
            screenWidth = monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left;
            screenHeight = monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top;
        }

        initialized = true;
        return true;
    }
    catch (...) {
        return false;
    }
}

void WGCScreenCapture::CleanupWGC() {
    StopCapture();
    CleanupTextureCache();

    if (frameTexture) {
        frameTexture->Release();
        frameTexture = nullptr;
    }

    if (pd3dContext) {
        pd3dContext->Release();
        pd3dContext = nullptr;
    }

    if (pd3dDevice) {
        pd3dDevice->Release();
        pd3dDevice = nullptr;
    }

    device = nullptr;
    captureItem = nullptr;

    initialized = false;
}

bool WGCScreenCapture::InitializeTextures() {
    if (textureInitialized) {
        CleanupTextureCache();
        textureInitialized = false;
    }

    D3D11_TEXTURE2D_DESC stagingDesc = {};
    stagingDesc.Width = captureWidth;
    stagingDesc.Height = captureHeight;
    stagingDesc.MipLevels = 1;
    stagingDesc.ArraySize = 1;
    stagingDesc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    stagingDesc.SampleDesc.Count = 1;
    stagingDesc.SampleDesc.Quality = 0;
    stagingDesc.Usage = D3D11_USAGE_STAGING;
    stagingDesc.BindFlags = 0;
    stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

    HRESULT hr = pd3dDevice->CreateTexture2D(&stagingDesc, nullptr, &texturePair_general.stagingTexture);
    if (FAILED(hr) || !texturePair_general.stagingTexture) {
        CleanupTextureCache();
        return false;
    }

    D3D11_QUERY_DESC queryDesc = {};
    queryDesc.Query = D3D11_QUERY_EVENT;
    queryDesc.MiscFlags = 0;

    hr = pd3dDevice->CreateQuery(&queryDesc, &texturePair_general.syncQuery);
    if (FAILED(hr)) {
        CleanupTextureCache();
        return false;
    }

    texturePair_general.width = captureWidth;
    texturePair_general.height = captureHeight;
    texturePair_general.inUse = false;
    texturePair_general.isMapped = false;

    textureInitialized = true;
    return true;
}

void WGCScreenCapture::CleanupTextureCache() {
    if (texturePair_general.isMapped && texturePair_general.stagingTexture && pd3dContext) {
        pd3dContext->Unmap(texturePair_general.stagingTexture, 0);
        texturePair_general.isMapped = false;
    }

    if (texturePair_general.stagingTexture) {
        texturePair_general.stagingTexture->Release();
        texturePair_general.stagingTexture = nullptr;
    }

    if (texturePair_general.syncQuery) {
        texturePair_general.syncQuery->Release();
        texturePair_general.syncQuery = nullptr;
    }

    textureInitialized = false;
}

CachedTexturePair* WGCScreenCapture::GetCachedTexturePair(UINT width, UINT height) {
    if (textureInitialized &&
        texturePair_general.stagingTexture &&
        texturePair_general.width == width &&
        texturePair_general.height == height &&
        !texturePair_general.inUse) {

        texturePair_general.inUse = true;
        return &texturePair_general;
    }
    return nullptr;
}

void WGCScreenCapture::RecalculateCenterPosition() {
    int centerX = (screenWidth - captureWidth) / 2;
    int centerY = (screenHeight - captureHeight) / 2;

    captureX = std::max(0, centerX);
    captureY = std::max(0, centerY);

    if (captureX + captureWidth > screenWidth) {
        captureWidth = screenWidth - captureX;
    }
    if (captureY + captureHeight > screenHeight) {
        captureHeight = screenHeight - captureY;
    }
}

bool WGCScreenCapture::UpdateFrame() {
    if (!isRunning || !framePool) return false;

    try {
        auto frame = framePool.TryGetNextFrame();
        if (!frame) {
            updated = false;
            return true;
        }

        ResizeIfNeeded();

        if (frameTexture) {
            frameTexture->Release();
            frameTexture = nullptr;
        }

        auto surface = GetDXGIInterfaceFromObject<ID3D11Texture2D>(frame.Surface());
        if (!surface) {
            updated = false;
            return false;
        }

        frameTexture = surface.detach();
        updated = true;
        return true;
    }
    catch (...) {
        updated = false;
        return false;
    }
}

void WGCScreenCapture::ReleaseFrame() {
}

void WGCScreenCapture::ConvertBGRA2BGR_SIMD(const uint8_t* bgra, uint8_t* bgr, int width, int height, int stride) {
    if (sse42Supported && width >= 4) {
        for (int y = 0; y < height; y++) {
            const uint8_t* srcRow = bgra + y * stride;
            uint8_t* destRow = bgr + y * width * 3;

            if (y + 1 < height) {
                _mm_prefetch((const char*)(bgra + (y + 1) * stride), _MM_HINT_T0);
            }

            int pixelX = 0;
            for (; pixelX <= width - 4; pixelX += 4) {
                __m128i pixels = _mm_loadu_si128((__m128i*)(srcRow + pixelX * 4));
                __m128i result = _mm_shuffle_epi8(pixels,
                    _mm_setr_epi8(
                        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                        0, 0, 0, 0
                    ));
                _mm_storeu_si128((__m128i*)(destRow + pixelX * 3), result);
            }

            for (; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }
    else {
        for (int y = 0; y < height; y++) {
            const uint8_t* srcRow = bgra + y * stride;
            uint8_t* destRow = bgr + y * width * 3;

            for (int pixelX = 0; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }
}

bool WGCScreenCapture::ConvertToBMP_Fast(const BYTE* srcData, int width, int height, int srcPitch, std::vector<uint8_t>& outBuffer) {
    if (!ValidateBMPParameters(srcData, width, height, srcPitch)) {
        return false;
    }

    BMPHeaderCache* headerCache = nullptr;
    {
        auto key = std::make_pair(width, height);
        auto it = bmpHeaderCache.find(key);
        if (it == bmpHeaderCache.end()) {
            bmpHeaderCache[key].Initialize(width, height);
        }
        headerCache = &bmpHeaderCache[key];
    }

    int totalSize = headerCache->totalSize;
    int rowSize = headerCache->rowSize;

    outBuffer.resize(totalSize);

    memcpy(outBuffer.data(), &headerCache->fileHeader, sizeof(BMPFileHeader));
    memcpy(outBuffer.data() + sizeof(BMPFileHeader), &headerCache->infoHeader, sizeof(BMPInfoHeader));

    BYTE* destData = outBuffer.data() + sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

    if (sse42Supported && width >= 4) {
        for (int y = 0; y < height; y++) {
            const BYTE* srcRow = srcData + (height - 1 - y) * srcPitch;
            BYTE* destRow = destData + y * rowSize;

            if (y + 1 < height) {
                _mm_prefetch((const char*)(srcData + (height - 1 - (y + 1)) * srcPitch), _MM_HINT_T0);
            }

            int pixelX = 0;
            for (; pixelX <= width - 4; pixelX += 4) {
                __m128i bgra = _mm_loadu_si128((__m128i*)(srcRow + pixelX * 4));
                __m128i bgr = _mm_shuffle_epi8(bgra,
                    _mm_setr_epi8(
                        0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14,
                        0, 0, 0, 0
                    ));
                _mm_storeu_si128((__m128i*)(destRow + pixelX * 3), bgr);
            }

            for (; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }
    else {
        for (int y = 0; y < height; y++) {
            const BYTE* srcRow = srcData + (height - 1 - y) * srcPitch;
            BYTE* destRow = destData + y * rowSize;

            for (int pixelX = 0; pixelX < width; pixelX++) {
                destRow[pixelX * 3 + 0] = srcRow[pixelX * 4 + 0];
                destRow[pixelX * 3 + 1] = srcRow[pixelX * 4 + 1];
                destRow[pixelX * 3 + 2] = srcRow[pixelX * 4 + 2];
            }
        }
    }

    return true;
}

winrt::com_ptr<ID3D11Device> WGCScreenCapture::CreateD3DDevice() {
    try {
        winrt::com_ptr<ID3D11Device> device;
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;

        D3D_FEATURE_LEVEL featureLevels[] = {
            D3D_FEATURE_LEVEL_11_1,
            D3D_FEATURE_LEVEL_11_0,
            D3D_FEATURE_LEVEL_10_1,
            D3D_FEATURE_LEVEL_10_0
        };

        HRESULT hr = D3D11CreateDevice(
            nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
            featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION,
            device.put(), nullptr, nullptr
        );

        if (DXGI_ERROR_UNSUPPORTED == hr) {
            hr = D3D11CreateDevice(
                nullptr, D3D_DRIVER_TYPE_WARP, nullptr, flags,
                featureLevels, ARRAYSIZE(featureLevels), D3D11_SDK_VERSION,
                device.put(), nullptr, nullptr
            );
        }

        return SUCCEEDED(hr) ? device : nullptr;
    }
    catch (...) {
        return nullptr;
    }
}

WGCScreenCapture::Direct3DDevice WGCScreenCapture::CreateDirect3DDevice() {
    try {
        winrt::com_ptr<IDXGIDevice> dxgiDevice;
        HRESULT hr = pd3dDevice->QueryInterface(__uuidof(IDXGIDevice), dxgiDevice.put_void());
        if (FAILED(hr)) return nullptr;

        winrt::com_ptr<::IInspectable> inspectable;
        hr = CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.get(), inspectable.put());
        if (FAILED(hr)) return nullptr;

        return inspectable.as<Direct3DDevice>();
    }
    catch (...) {
        return nullptr;
    }
}

WGCScreenCapture::CaptureItem WGCScreenCapture::CreateCaptureItemForWindow(HWND hwnd) {
    try {
        auto factory = winrt::get_activation_factory<CaptureItem>();
        auto interopFactory = factory.as<IGraphicsCaptureItemInterop>();
        CaptureItem item = nullptr;

        HRESULT hr = interopFactory->CreateForWindow(
            hwnd,
            winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
            winrt::put_abi(item));

        return SUCCEEDED(hr) ? item : nullptr;
    }
    catch (...) {
        return nullptr;
    }
}

WGCScreenCapture::CaptureItem WGCScreenCapture::CreateCaptureItemForMonitor(HMONITOR hmonitor) {
    try {
        auto factory = winrt::get_activation_factory<CaptureItem>();
        auto interopFactory = factory.as<IGraphicsCaptureItemInterop>();
        CaptureItem item = nullptr;

        HRESULT hr = interopFactory->CreateForMonitor(
            hmonitor,
            winrt::guid_of<ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
            winrt::put_abi(item));

        return SUCCEEDED(hr) ? item : nullptr;
    }
    catch (...) {
        return nullptr;
    }
}

bool WGCScreenCapture::StartCapture() {
    try {
        if (!captureItem || !device) return false;

        auto itemSize = captureItem.Size();
        lastSize = itemSize;

        framePool = winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool::
            CreateFreeThreaded(device, pixelFormat, 1, itemSize);
        if (!framePool) return false;

        captureSession = framePool.CreateCaptureSession(captureItem);
        if (!captureSession) return false;

        captureSession.IsCursorCaptureEnabled(false);
        captureSession.StartCapture();
        isRunning = true;

        return true;
    }
    catch (...) {
        return false;
    }
}

void WGCScreenCapture::StopCapture() {
    try {
        isRunning = false;

        if (captureSession) {
            captureSession.Close();
            captureSession = nullptr;
        }

        if (framePool) {
            framePool.Close();
            framePool = nullptr;
        }
    }
    catch (...) {}
}

bool WGCScreenCapture::ResizeIfNeeded() {
    try {
        if (!captureItem) return false;

        auto contentSize = captureItem.Size();
        if ((contentSize.Width != lastSize.Width) || (contentSize.Height != lastSize.Height)) {
            lastSize = contentSize;

            screenWidth = contentSize.Width;
            screenHeight = contentSize.Height;

            if (useCenterMode) {
                RecalculateCenterPosition();
            }

            if (framePool) {
                framePool.Recreate(device, pixelFormat, 1, lastSize);
            }

            return true;
        }

        return false;
    }
    catch (...) {
        return false;
    }
}

bool WGCScreenCapture::Initialize(int width, int height) {
    Release();

    if (!InitializeWGC()) {
        return false;
    }

    useCenterMode = true;

    int centerX = (screenWidth - width) / 2;
    int centerY = (screenHeight - height) / 2;

    centerX = std::max(0, centerX);
    centerY = std::max(0, centerY);

    captureX = centerX;
    captureY = centerY;
    captureWidth = std::min(width, screenWidth - centerX);
    captureHeight = std::min(height, screenHeight - centerY);

    bool isNewResolution = true;
    for (auto& rc : resolutionFirstCalls) {
        if (rc.width == captureWidth && rc.height == captureHeight) {
            isNewResolution = false;
            break;
        }
    }
    if (isNewResolution) {
        resolutionFirstCalls.emplace_back(captureWidth, captureHeight);
    }

    if (!InitializeTextures()) {
        return false;
    }

    outWidth = captureWidth;
    outHeight = captureHeight;
    outChannels = 3;

    hmonitor = MonitorFromPoint({ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
    captureItem = CreateCaptureItemForMonitor(hmonitor);
    if (!captureItem) return false;

    if (!StartCapture()) return false;

    return true;
}

bool WGCScreenCapture::InitializeRegion(int x, int y, int width, int height) {
    Release();

    if (!InitializeWGC()) {
        return false;
    }

    useCenterMode = false;

    captureX = std::max(0, std::min(x, screenWidth));
    captureY = std::max(0, std::min(y, screenHeight));
    captureWidth = std::max(1, std::min(width, screenWidth - captureX));
    captureHeight = std::max(1, std::min(height, screenHeight - captureY));

    bool isNewResolution = true;
    for (auto& rc : resolutionFirstCalls) {
        if (rc.width == captureWidth && rc.height == captureHeight) {
            isNewResolution = false;
            break;
        }
    }
    if (isNewResolution) {
        resolutionFirstCalls.emplace_back(captureWidth, captureHeight);
    }

    if (!InitializeTextures()) {
        return false;
    }

    outWidth = captureWidth;
    outHeight = captureHeight;
    outChannels = 3;

    hmonitor = MonitorFromPoint({ 0, 0 }, MONITOR_DEFAULTTOPRIMARY);
    captureItem = CreateCaptureItemForMonitor(hmonitor);
    if (!captureItem) return false;

    if (!StartCapture()) return false;

    return true;
}

bool WGCScreenCapture::SetRegion(int x, int y, int width, int height) {
    return InitializeRegion(x, y, width, height);
}

bool WGCScreenCapture::SetWindow(HWND hwnd) {
    StopCapture();

    monitorMode = (hwnd == NULL);
    targetWindow = hwnd;

    try {
        if (monitorMode) {
            captureItem = CreateCaptureItemForMonitor(hmonitor);
        }
        else {
            captureItem = CreateCaptureItemForWindow(hwnd);
        }

        if (!captureItem) return false;
        if (!StartCapture()) return false;

        return true;
    }
    catch (...) {
        return false;
    }
}

std::vector<uint8_t> WGCScreenCapture::CaptureBGR() {
    std::vector<uint8_t> result;

    if (!initialized || !pd3dDevice || !pd3dContext || !isRunning) {
        return result;
    }

    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized || !pd3dDevice || !pd3dContext || !isRunning) {
            return result;
        }
    }

    int loopCount = 0;
    const int MAX_RETRY = 10;

    while (loopCount < MAX_RETRY) {
        loopCount++;

        if (!UpdateFrame()) {
            return result;
        }

        if (!updated || !frameTexture) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        break;
    }

    if (loopCount >= MAX_RETRY && (!updated || !frameTexture)) {
        return result;
    }

    CachedTexturePair* pTexturePair = GetCachedTexturePair(captureWidth, captureHeight);
    if (!pTexturePair || !pTexturePair->stagingTexture) {
        if (pTexturePair) pTexturePair->inUse = false;
        return result;
    }

    D3D11_TEXTURE2D_DESC frameDesc;
    frameTexture->GetDesc(&frameDesc);

    int frameW = static_cast<int>(frameDesc.Width);
    int frameH = static_cast<int>(frameDesc.Height);
    int startX = std::max(0, std::min(captureX, frameW - 1));
    int startY = std::max(0, std::min(captureY, frameH - 1));
    int actualWidth = std::min(captureWidth, frameW - startX);
    int actualHeight = std::min(captureHeight, frameH - startY);

    if (actualWidth <= 0 || actualHeight <= 0) {
        pTexturePair->inUse = false;
        return result;
    }

    D3D11_BOX sourceBox = {
        (UINT)startX, (UINT)startY, 0,
        (UINT)(startX + actualWidth), (UINT)(startY + actualHeight), 1
    };

    pd3dContext->CopySubresourceRegion(
        pTexturePair->stagingTexture, 0, 0, 0, 0,
        frameTexture, 0, &sourceBox
    );

    pd3dContext->End(pTexturePair->syncQuery);
    ReleaseFrame();

    BOOL queryData = FALSE;
    while (pd3dContext->GetData(pTexturePair->syncQuery, &queryData, sizeof(BOOL), 0) == S_FALSE) {
        std::this_thread::yield();
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (pTexturePair->isMapped) {
        pd3dContext->Unmap(pTexturePair->stagingTexture, 0);
        pTexturePair->isMapped = false;
    }

    HRESULT hr = pd3dContext->Map(
        pTexturePair->stagingTexture, 0,
        D3D11_MAP_READ,
        0,
        &mappedResource
    );

    if (FAILED(hr)) {
        pTexturePair->inUse = false;
        return result;
    }

    pTexturePair->isMapped = true;
    pTexturePair->mappedResource = mappedResource;

    size_t bgrSize = actualWidth * actualHeight * 3;
    result.resize(bgrSize);

    ConvertBGRA2BGR_SIMD(
        (const uint8_t*)mappedResource.pData,
        result.data(),
        actualWidth, actualHeight,
        mappedResource.RowPitch
    );

    pTexturePair->inUse = false;

    return result;
}

bool WGCScreenCapture::CaptureBGR(uint8_t* outBuffer, size_t bufferSize) {
    if (!outBuffer || !initialized || !pd3dDevice || !pd3dContext || !isRunning) {
        return false;
    }

    size_t requiredSize = (size_t)captureWidth * captureHeight * 3;
    if (bufferSize < requiredSize) return false;

    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized || !pd3dDevice || !pd3dContext || !isRunning) {
            return false;
        }
    }

    int loopCount = 0;
    const int MAX_RETRY = 10;

    while (loopCount < MAX_RETRY) {
        loopCount++;
        if (!UpdateFrame()) return false;
        if (!updated || !frameTexture) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
            continue;
        }
        break;
    }

    if (loopCount >= MAX_RETRY && (!updated || !frameTexture)) {
        return false;
    }

    CachedTexturePair* pTexturePair = GetCachedTexturePair(captureWidth, captureHeight);
    if (!pTexturePair || !pTexturePair->stagingTexture) {
        if (pTexturePair) pTexturePair->inUse = false;
        return false;
    }

    D3D11_TEXTURE2D_DESC frameDesc;
    frameTexture->GetDesc(&frameDesc);

    int frameW = static_cast<int>(frameDesc.Width);
    int frameH = static_cast<int>(frameDesc.Height);
    int startX = std::max(0, std::min(captureX, frameW - 1));
    int startY = std::max(0, std::min(captureY, frameH - 1));
    int actualWidth = std::min(captureWidth, frameW - startX);
    int actualHeight = std::min(captureHeight, frameH - startY);

    if (actualWidth <= 0 || actualHeight <= 0) {
        pTexturePair->inUse = false;
        return false;
    }

    D3D11_BOX sourceBox = {
        (UINT)startX, (UINT)startY, 0,
        (UINT)(startX + actualWidth), (UINT)(startY + actualHeight), 1
    };

    pd3dContext->CopySubresourceRegion(
        pTexturePair->stagingTexture, 0, 0, 0, 0,
        frameTexture, 0, &sourceBox
    );

    pd3dContext->End(pTexturePair->syncQuery);
    ReleaseFrame();

    BOOL queryData = FALSE;
    while (pd3dContext->GetData(pTexturePair->syncQuery, &queryData, sizeof(BOOL), 0) == S_FALSE) {
        std::this_thread::yield();
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (pTexturePair->isMapped) {
        pd3dContext->Unmap(pTexturePair->stagingTexture, 0);
        pTexturePair->isMapped = false;
    }

    HRESULT hr = pd3dContext->Map(
        pTexturePair->stagingTexture, 0,
        D3D11_MAP_READ, 0, &mappedResource
    );

    if (FAILED(hr)) {
        pTexturePair->inUse = false;
        return false;
    }

    pTexturePair->isMapped = true;
    pTexturePair->mappedResource = mappedResource;

    ConvertBGRA2BGR_SIMD(
        (const uint8_t*)mappedResource.pData,
        outBuffer,
        actualWidth, actualHeight,
        mappedResource.RowPitch
    );

    pTexturePair->inUse = false;
    return true;
}

std::vector<uint8_t> WGCScreenCapture::CaptureBMP() {
    std::vector<uint8_t> result;

    if (!initialized || !pd3dDevice || !pd3dContext || !isRunning) {
        return result;
    }

    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized || !pd3dDevice || !pd3dContext || !isRunning) {
            return result;
        }
    }

    int loopCount = 0;
    const int MAX_RETRY = 10;

    while (loopCount < MAX_RETRY) {
        loopCount++;

        if (!UpdateFrame()) {
            return result;
        }

        if (!updated || !frameTexture) {
            continue;
        }
        break;
    }

    if (loopCount >= MAX_RETRY && (!updated || !frameTexture)) {
        return result;
    }

    CachedTexturePair* pTexturePair = GetCachedTexturePair(captureWidth, captureHeight);
    if (!pTexturePair || !pTexturePair->stagingTexture) {
        if (pTexturePair) pTexturePair->inUse = false;
        return result;
    }

    D3D11_TEXTURE2D_DESC frameDesc;
    frameTexture->GetDesc(&frameDesc);

    int frameW = static_cast<int>(frameDesc.Width);
    int frameH = static_cast<int>(frameDesc.Height);
    int startX = std::max(0, std::min(captureX, frameW - 1));
    int startY = std::max(0, std::min(captureY, frameH - 1));
    int actualWidth = std::min(captureWidth, frameW - startX);
    int actualHeight = std::min(captureHeight, frameH - startY);

    if (actualWidth <= 0 || actualHeight <= 0) {
        pTexturePair->inUse = false;
        return result;
    }

    D3D11_BOX sourceBox = {
        (UINT)startX, (UINT)startY, 0,
        (UINT)(startX + actualWidth), (UINT)(startY + actualHeight), 1
    };

    pd3dContext->CopySubresourceRegion(
        pTexturePair->stagingTexture, 0, 0, 0, 0,
        frameTexture, 0, &sourceBox
    );

    pd3dContext->End(pTexturePair->syncQuery);
    ReleaseFrame();

    BOOL queryData = FALSE;
    while (pd3dContext->GetData(pTexturePair->syncQuery, &queryData, sizeof(BOOL), 0) == S_FALSE) {
        std::this_thread::yield();
    }

    D3D11_MAPPED_SUBRESOURCE mappedResource;
    if (pTexturePair->isMapped) {
        pd3dContext->Unmap(pTexturePair->stagingTexture, 0);
        pTexturePair->isMapped = false;
    }

    HRESULT hr = pd3dContext->Map(
        pTexturePair->stagingTexture, 0,
        D3D11_MAP_READ,
        0,
        &mappedResource
    );

    if (FAILED(hr)) {
        pTexturePair->inUse = false;
        return result;
    }

    pTexturePair->isMapped = true;
    pTexturePair->mappedResource = mappedResource;

    bool success = ConvertToBMP_Fast(
        (const BYTE*)mappedResource.pData,
        actualWidth, actualHeight,
        mappedResource.RowPitch,
        result
    );

    pTexturePair->inUse = false;

    if (!success) {
        result.clear();
    }

    return result;
}

void WGCScreenCapture::Release() {
    CleanupWGC();

    bmpHeaderCache.clear();
    resolutionFirstCalls.clear();
    frameCounter = 0;
    updated = false;
}
