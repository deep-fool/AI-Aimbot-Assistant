#include "GDIScreenCapture.h"
#include <algorithm>
#include <intrin.h>

GDIResource::~GDIResource() {
    Release();
}

bool GDIResource::Initialize(HWND target, int width, int height) {
    Release();
    targetWindow = target;

    screenDC = GetDC(nullptr);
    if (!screenDC) return false;

    if (targetWindow) {
        windowDC = GetDC(targetWindow);
        if (!windowDC) {
            ReleaseDC(nullptr, screenDC);
            screenDC = nullptr;
            return false;
        }
    }
    else {
        windowDC = screenDC;
    }

    memoryDC = CreateCompatibleDC(windowDC);
    if (!memoryDC) {
        if (windowDC != screenDC) ReleaseDC(targetWindow, windowDC);
        ReleaseDC(nullptr, screenDC);
        screenDC = windowDC = nullptr;
        return false;
    }

    bitmap = CreateCompatibleBitmap(windowDC, width, height);
    if (!bitmap) {
        DeleteDC(memoryDC);
        if (windowDC != screenDC) ReleaseDC(targetWindow, windowDC);
        ReleaseDC(nullptr, screenDC);
        screenDC = windowDC = memoryDC = nullptr;
        return false;
    }

    SelectObject(memoryDC, bitmap);
    return true;
}

void GDIResource::Release() {
    if (bitmap) {
        DeleteObject(bitmap);
        bitmap = nullptr;
    }
    if (memoryDC) {
        DeleteDC(memoryDC);
        memoryDC = nullptr;
    }
    if (windowDC && windowDC != screenDC) {
        ReleaseDC(targetWindow, windowDC);
        windowDC = nullptr;
    }
    if (screenDC) {
        ReleaseDC(nullptr, screenDC);
        screenDC = nullptr;
    }
    targetWindow = nullptr;
}

GDIScreenCapture::GDIScreenCapture()
    : sse42Supported(false)
    , captureTimeoutMs(0)
    , useCenterMode(false)
    , lastScreenWidth(0)
    , lastScreenHeight(0)
    , originalWidth(0)
    , originalHeight(0) {

    sse42Supported = IsSSE42Supported();

    ZeroMemory(&bitmapInfo, sizeof(BITMAPINFOHEADER));
    bitmapInfo.biSize = sizeof(BITMAPINFOHEADER);
    bitmapInfo.biPlanes = 1;
    bitmapInfo.biBitCount = 24;
    bitmapInfo.biCompression = BI_RGB;
}

GDIScreenCapture::~GDIScreenCapture() {
    Release();
}

bool GDIScreenCapture::IsSSE42Supported() {
    int cpuInfo[4];
    __cpuid(cpuInfo, 1);
    return (cpuInfo[2] & (1 << 20)) != 0;
}

inline bool GDIScreenCapture::ValidateBMPParameters(const BYTE* srcData, int x, int y, int width, int height, int srcPitch) {
    return srcData && width > 0 && height > 0 && srcPitch > 0 && x >= 0 && y >= 0;
}

inline bool GDIScreenCapture::IsResolutionFirstCall(int width, int height) {
    for (auto& rc : resolutionFirstCalls) {
        if (rc.IsFirstCall(width, height)) {
            return true;
        }
    }
    return false;
}

void GDIScreenCapture::RecalculateCenterPosition() {
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    captureX = (screenWidth - originalWidth) / 2;
    captureY = (screenHeight - originalHeight) / 2;
    captureX = std::max(0, captureX);
    captureY = std::max(0, captureY);

    captureWidth = originalWidth;
    captureHeight = originalHeight;

    if (captureX + captureWidth > screenWidth) {
        captureWidth = screenWidth - captureX;
    }
    if (captureY + captureHeight > screenHeight) {
        captureHeight = screenHeight - captureY;
    }
}

void GDIScreenCapture::CheckAndUpdateScreenResolution() {
    if (!useCenterMode) return;

    int currentScreenWidth = GetSystemMetrics(SM_CXSCREEN);
    int currentScreenHeight = GetSystemMetrics(SM_CYSCREEN);

    if (currentScreenWidth != lastScreenWidth || currentScreenHeight != lastScreenHeight) {
        lastScreenWidth = currentScreenWidth;
        lastScreenHeight = currentScreenHeight;

        RecalculateCenterPosition();
    }
}

bool GDIScreenCapture::Initialize(int width, int height) {
    Release();

    captureWidth = width;
    captureHeight = height;
    originalWidth = width;
    originalHeight = height;
    useCustomRegion = false;
    useCenterMode = true;

    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    lastScreenWidth = screenWidth;
    lastScreenHeight = screenHeight;

    captureX = (screenWidth - width) / 2;
    captureY = (screenHeight - height) / 2;
    captureX = std::max(0, captureX);
    captureY = std::max(0, captureY);

    if (captureX + width > screenWidth) {
        captureWidth = screenWidth - captureX;
    }
    if (captureY + height > screenHeight) {
        captureHeight = screenHeight - captureY;
    }

    if (!gdiResources.Initialize(nullptr, captureWidth, captureHeight)) {
        return false;
    }

    bitmapInfo.biWidth = captureWidth;
    bitmapInfo.biHeight = captureHeight;

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

    int rowStride = ((captureWidth * 3 + 3) & ~3);
    dibBuffer.resize(rowStride * captureHeight);

    initialized = true;
    return true;
}

bool GDIScreenCapture::InitializeRegion(int x, int y, int width, int height) {
    Release();

    captureX = std::max(0, x);
    captureY = std::max(0, y);
    captureWidth = width;
    captureHeight = height;
    originalWidth = width;
    originalHeight = height;
    useCustomRegion = true;
    useCenterMode = false;

    if (!gdiResources.Initialize(nullptr, width, height)) {
        return false;
    }

    bitmapInfo.biWidth = width;
    bitmapInfo.biHeight = height;

    bool isNewResolution = true;
    for (auto& rc : resolutionFirstCalls) {
        if (rc.width == width && rc.height == height) {
            isNewResolution = false;
            break;
        }
    }
    if (isNewResolution) {
        resolutionFirstCalls.emplace_back(width, height);
    }

    int rowStride = ((width * 3 + 3) & ~3);
    dibBuffer.resize(rowStride * height);

    initialized = true;
    return true;
}

bool GDIScreenCapture::SetRegion(int x, int y, int width, int height) {
    if (!initialized) return false;

    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);

    x = std::max(0, x);
    y = std::max(0, y);

    if (x + width > screenWidth) width = screenWidth - x;
    if (y + height > screenHeight) height = screenHeight - y;
    if (width <= 0 || height <= 0) return false;

    captureX = x;
    captureY = y;
    captureWidth = width;
    captureHeight = height;
    useCustomRegion = true;
    useCenterMode = false;

    if (!UpdateBitmapIfNeeded(width, height)) {
        return false;
    }

    bitmapInfo.biWidth = width;
    bitmapInfo.biHeight = height;

    int rowStride = ((width * 3 + 3) & ~3);
    dibBuffer.resize(rowStride * height);

    return true;
}

bool GDIScreenCapture::UpdateBitmapIfNeeded(int width, int height) {
    BITMAP bitmapStruct;
    if (!GetObject(gdiResources.bitmap, sizeof(BITMAP), &bitmapStruct)) {
        return false;
    }

    if (bitmapStruct.bmWidth == width && bitmapStruct.bmHeight == height) {
        return true;
    }

    if (gdiResources.bitmap) {
        DeleteObject(gdiResources.bitmap);
        gdiResources.bitmap = nullptr;
    }

    gdiResources.bitmap = CreateCompatibleBitmap(gdiResources.windowDC, width, height);
    if (!gdiResources.bitmap) return false;

    SelectObject(gdiResources.memoryDC, gdiResources.bitmap);
    return true;
}

bool GDIScreenCapture::SetWindow(HWND hwnd) {
    if (gdiResources.targetWindow != hwnd) {
        gdiResources.targetWindow = hwnd;

        if (initialized && hwnd) {
            RECT clientRect;
            if (GetClientRect(hwnd, &clientRect)) {
                int windowWidth = clientRect.right - clientRect.left;
                int windowHeight = clientRect.bottom - clientRect.top;

                if (useCenterMode) {
                    captureX = (windowWidth - captureWidth) / 2;
                    captureY = (windowHeight - captureHeight) / 2;
                    captureX = std::max(0, captureX);
                    captureY = std::max(0, captureY);

                    if (captureX + captureWidth > windowWidth) {
                        captureWidth = windowWidth - captureX;
                    }
                    if (captureY + captureHeight > windowHeight) {
                        captureHeight = windowHeight - captureY;
                    }
                }
            }

            return gdiResources.Initialize(hwnd, captureWidth, captureHeight);
        }
    }

    return true;
}

void GDIScreenCapture::Release() {
    initialized = false;
    gdiResources.Release();

    bmpHeaderCache.clear();
    resolutionFirstCalls.clear();

    dibBuffer.clear();
    dibBuffer.shrink_to_fit();

    captureX = captureY = 0;
    captureWidth = captureHeight = 0;
    useCustomRegion = false;
    useCenterMode = false;
    lastScreenWidth = 0;
    lastScreenHeight = 0;
    originalWidth = 0;
    originalHeight = 0;
}

bool GDIScreenCapture::PerformCapture() {
    if (!initialized) return false;

    HDC sourceDC = gdiResources.windowDC;

    BOOL result = BitBlt(
        gdiResources.memoryDC, 0, 0, captureWidth, captureHeight,
        sourceDC, captureX, captureY, SRCCOPY
    );

    return result != FALSE;
}

void GDIScreenCapture::ConvertDIBtoBGR_SIMD(const uint8_t* src, uint8_t* dst, int width, int height, int srcStride) {
    const int dstStride = width * 3;

    if (sse42Supported && width >= 8) {
        for (int y = 0; y < height; y++) {
            const uint8_t* srcRow = src + (height - 1 - y) * srcStride;
            uint8_t* dstRow = dst + y * dstStride;

            if (y + 1 < height) {
                _mm_prefetch((const char*)(src + (height - 2 - y) * srcStride), _MM_HINT_T0);
            }

            int x = 0;
            for (; x + 7 < width; x += 8) {
                __m128i chunk1 = _mm_loadu_si128((const __m128i*)(srcRow + x * 3));
                __m128i chunk2 = _mm_loadl_epi64((const __m128i*)(srcRow + x * 3 + 16));

                _mm_storeu_si128((__m128i*)(dstRow + x * 3), chunk1);
                _mm_storel_epi64((__m128i*)(dstRow + x * 3 + 16), chunk2);
            }

            for (; x < width; x++) {
                dstRow[x * 3 + 0] = srcRow[x * 3 + 0];
                dstRow[x * 3 + 1] = srcRow[x * 3 + 1];
                dstRow[x * 3 + 2] = srcRow[x * 3 + 2];
            }
        }
    }
    else {
        for (int y = 0; y < height; y++) {
            const uint8_t* srcRow = src + (height - 1 - y) * srcStride;
            uint8_t* dstRow = dst + y * dstStride;
            memcpy(dstRow, srcRow, width * 3);
        }
    }
}

bool GDIScreenCapture::ExtractBGRData(std::vector<uint8_t>& outBuffer) {
    int rowStride = ((captureWidth * 3 + 3) & ~3);

    int result = GetDIBits(
        gdiResources.windowDC,
        gdiResources.bitmap,
        0, captureHeight,
        dibBuffer.data(),
        reinterpret_cast<BITMAPINFO*>(&bitmapInfo),
        DIB_RGB_COLORS
    );

    if (result <= 0) return false;

    // 调整输出 buffer 大小
    size_t bgrSize = captureWidth * captureHeight * 3;
    outBuffer.resize(bgrSize);

    ConvertDIBtoBGR_SIMD(
        dibBuffer.data(),
        outBuffer.data(),
        captureWidth, captureHeight,
        rowStride
    );

    return true;
}

bool GDIScreenCapture::ExtractBMPData(std::vector<uint8_t>& outBuffer) {
    int rowStride = ((captureWidth * 3 + 3) & ~3);
    DWORD bmpDataSize = rowStride * captureHeight;
    DWORD headerSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
    DWORD totalSize = headerSize + bmpDataSize;

    // 调整输出 buffer 大小
    outBuffer.resize(totalSize);

    // 填充 BMP 文件头
    BMPFileHeader* fileHeader = (BMPFileHeader*)outBuffer.data();
    fileHeader->bfType = 0x4D42;
    fileHeader->bfSize = totalSize;
    fileHeader->bfReserved1 = 0;
    fileHeader->bfReserved2 = 0;
    fileHeader->bfOffBits = headerSize;

    // 填充 BMP 信息头
    BMPInfoHeader* infoHeader = (BMPInfoHeader*)(outBuffer.data() + sizeof(BMPFileHeader));
    memcpy(infoHeader, &bitmapInfo, sizeof(BMPInfoHeader));

    // 获取位图数据
    int result = GetDIBits(
        gdiResources.windowDC,
        gdiResources.bitmap,
        0, captureHeight,
        outBuffer.data() + headerSize,
        reinterpret_cast<BITMAPINFO*>(&bitmapInfo),
        DIB_RGB_COLORS
    );

    return (result > 0);
}

std::vector<uint8_t> GDIScreenCapture::CaptureBGR() {
    std::vector<uint8_t> result;

    if (!initialized) return result;

    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized) return result;
    }

    CheckAndUpdateScreenResolution();

    if (!PerformCapture()) return result;
    if (!ExtractBGRData(result)) {
        result.clear();
        return result;
    }

    return result;
}

bool GDIScreenCapture::CaptureBGR(uint8_t* outBuffer, size_t bufferSize) {
    if (!outBuffer || !initialized) return false;

    size_t requiredSize = (size_t)captureWidth * captureHeight * 3;
    if (bufferSize < requiredSize) return false;

    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized) return false;
    }

    CheckAndUpdateScreenResolution();

    if (!PerformCapture()) return false;

    int rowStride = ((captureWidth * 3 + 3) & ~3);

    int result = GetDIBits(
        gdiResources.windowDC,
        gdiResources.bitmap,
        0, captureHeight,
        dibBuffer.data(),
        reinterpret_cast<BITMAPINFO*>(&bitmapInfo),
        DIB_RGB_COLORS
    );

    if (result <= 0) return false;

    ConvertDIBtoBGR_SIMD(
        dibBuffer.data(),
        outBuffer,
        captureWidth, captureHeight,
        rowStride
    );

    return true;
}

std::vector<uint8_t> GDIScreenCapture::CaptureBMP() {
    std::vector<uint8_t> result;

    if (!initialized) return result;

    if (IsResolutionFirstCall(captureWidth, captureHeight)) {
        if (!initialized) return result;
    }

    CheckAndUpdateScreenResolution();

    if (!PerformCapture()) return result;
    if (!ExtractBMPData(result)) {
        result.clear();
        return result;
    }

    return result;
}
