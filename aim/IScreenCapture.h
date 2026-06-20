#ifndef ISCREEN_CAPTURE_H
#define ISCREEN_CAPTURE_H

#include <Windows.h>
#include <cstdint>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <vector>
#include <memory>
#include <unordered_map>

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")

//============================================================================
// BMP 文件头结构
//============================================================================
#pragma pack(push, 1)
struct BMPFileHeader {
    WORD bfType;
    DWORD bfSize;
    WORD bfReserved1;
    WORD bfReserved2;
    DWORD bfOffBits;
};

struct BMPInfoHeader {
    DWORD biSize;
    LONG biWidth;
    LONG biHeight;
    WORD biPlanes;
    WORD biBitCount;
    DWORD biCompression;
    DWORD biSizeImage;
    LONG biXPelsPerMeter;
    LONG biYPelsPerMeter;
    DWORD biClrUsed;
    DWORD biClrImportant;
};
#pragma pack(pop)

//============================================================================
// 预分配内存池
//============================================================================
class PreallocatedMemoryPool {
public:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool inUse;

        MemoryBlock(void* p, size_t s) : ptr(p), size(s), inUse(false) {}
    };

    std::vector<MemoryBlock> regionBlocks;
    std::vector<MemoryBlock> bmpBlocks;
    void* currentRegionData;
    void* currentBmpData;
    size_t currentRegionSize;
    size_t currentBmpSize;

    PreallocatedMemoryPool() : currentRegionData(nullptr), currentBmpData(nullptr),
        currentRegionSize(0), currentBmpSize(0) {
    }

    bool Initialize(int maxWidth, int maxHeight) {
        size_t maxRegionSize = maxWidth * maxHeight * 4;
        size_t maxBmpSize = maxWidth * maxHeight * 3 + 1024;

        std::vector<size_t> regionSizes = {
            128 * 128 * 4, 160 * 120 * 4, 240 * 180 * 4, 320 * 240 * 4,
            320 * 320 * 4, 400 * 300 * 4, 480 * 360 * 4, 640 * 480 * 4,
            640 * 640 * 4, 800 * 600 * 4, 1024 * 768 * 4, 1280 * 720 * 4,
            1366 * 768 * 4, 1920 * 1080 * 4, maxRegionSize
        };

        std::vector<size_t> bmpSizes = {
            128 * 128 * 3 + 1024, 160 * 120 * 3 + 1024, 240 * 180 * 3 + 1024,
            320 * 240 * 3 + 1024, 320 * 320 * 3 + 1024, 400 * 300 * 3 + 1024,
            480 * 360 * 3 + 1024, 640 * 480 * 3 + 1024, 640 * 640 * 3 + 1024,
            800 * 600 * 3 + 1024, 1024 * 768 * 3 + 1024, 1280 * 720 * 3 + 1024,
            1366 * 768 * 3 + 1024, 1920 * 1080 * 3 + 1024, maxBmpSize
        };

        for (size_t size : regionSizes) {
            void* block = _aligned_malloc(size, 32);
            if (block) regionBlocks.emplace_back(block, size);
        }

        for (size_t size : bmpSizes) {
            void* block = _aligned_malloc(size, 32);
            if (block) bmpBlocks.emplace_back(block, size);
        }

        return !regionBlocks.empty() && !bmpBlocks.empty();
    }

    void* GetRegionBlock(size_t requiredSize) {
        for (auto& block : regionBlocks) {
            if (!block.inUse && block.size >= requiredSize) {
                block.inUse = true;
                currentRegionData = block.ptr;
                currentRegionSize = block.size;
                return block.ptr;
            }
        }
        return nullptr;
    }

    void* GetBmpBlock(size_t requiredSize) {
        for (auto& block : bmpBlocks) {
            if (!block.inUse && block.size >= requiredSize) {
                block.inUse = true;
                currentBmpData = block.ptr;
                currentBmpSize = block.size;
                return block.ptr;
            }
        }
        return nullptr;
    }

    void ReleaseCurrent() {
        if (currentRegionData) {
            for (auto& block : regionBlocks) {
                if (block.ptr == currentRegionData) {
                    block.inUse = false;
                    break;
                }
            }
            currentRegionData = nullptr;
        }

        if (currentBmpData) {
            for (auto& block : bmpBlocks) {
                if (block.ptr == currentBmpData) {
                    block.inUse = false;
                    break;
                }
            }
            currentBmpData = nullptr;
        }
    }

    ~PreallocatedMemoryPool() {
        for (auto& block : regionBlocks) {
            if (block.ptr) _aligned_free(block.ptr);
        }
        for (auto& block : bmpBlocks) {
            if (block.ptr) _aligned_free(block.ptr);
        }
    }
};

//============================================================================
// 纹理缓存对
//============================================================================
struct CachedTexturePair {
    ID3D11Texture2D* stagingTexture;
    bool inUse;
    UINT width;
    UINT height;
    bool isMapped;
    D3D11_MAPPED_SUBRESOURCE mappedResource;
    ID3D11Query* syncQuery;

    CachedTexturePair() : stagingTexture(nullptr), inUse(false), width(0), height(0),
        isMapped(false), syncQuery(nullptr) {
    }

    HRESULT MapTexture(ID3D11DeviceContext* pContext) {
        if (!isMapped && stagingTexture) {
            HRESULT hr = pContext->Map(stagingTexture, 0, D3D11_MAP_READ, 0, &mappedResource);
            if (SUCCEEDED(hr)) {
                isMapped = true;
            }
            return hr;
        }
        return S_OK;
    }

    void UnmapTexture(ID3D11DeviceContext* pContext) {
        if (isMapped && stagingTexture) {
            pContext->Unmap(stagingTexture, 0);
            isMapped = false;
        }
    }

    HRESULT RemapTexture(ID3D11DeviceContext* pContext) {
        if (isMapped) {
            pContext->Unmap(stagingTexture, 0);
            isMapped = false;
        }
        return MapTexture(pContext);
    }
};

//============================================================================
// BMP头部缓存
//============================================================================
struct BMPHeaderCache {
    BMPFileHeader fileHeader;
    BMPInfoHeader infoHeader;
    int totalSize;
    int imageSize;
    int rowSize;

    void Initialize(int width, int height) {
        rowSize = ((width * 3 + 3) / 4) * 4;
        imageSize = rowSize * height;
        totalSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader) + imageSize;

        if (totalSize < sizeof(BMPFileHeader) + sizeof(BMPInfoHeader)) {
            totalSize = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);
        }

        fileHeader.bfType = 0x4D42;
        fileHeader.bfSize = totalSize;
        fileHeader.bfReserved1 = 0;
        fileHeader.bfReserved2 = 0;
        fileHeader.bfOffBits = sizeof(BMPFileHeader) + sizeof(BMPInfoHeader);

        infoHeader.biSize = sizeof(BMPInfoHeader);
        infoHeader.biWidth = width;
        infoHeader.biHeight = height;
        infoHeader.biPlanes = 1;
        infoHeader.biBitCount = 24;
        infoHeader.biCompression = 0;
        infoHeader.biSizeImage = imageSize;
        infoHeader.biXPelsPerMeter = 0;
        infoHeader.biYPelsPerMeter = 0;
        infoHeader.biClrUsed = 0;
        infoHeader.biClrImportant = 0;
    }
};

struct PairHash {
    template <class T1, class T2>
    std::size_t operator () (const std::pair<T1, T2>& p) const {
        auto h1 = std::hash<T1>{}(p.first);
        auto h2 = std::hash<T2>{}(p.second);
        return h1 ^ (h2 << 1);
    }
};

//============================================================================
// 分辨率首次调用标志
//============================================================================
struct ResolutionFirstCall {
    int width;
    int height;
    bool firstCall;

    ResolutionFirstCall(int w, int h) : width(w), height(h), firstCall(true) {}

    bool IsFirstCall(int w, int h) {
        if (width == w && height == h && firstCall) {
            firstCall = false;
            return true;
        }
        return false;
    }
};

#endif // ISCREEN_CAPTURE_H