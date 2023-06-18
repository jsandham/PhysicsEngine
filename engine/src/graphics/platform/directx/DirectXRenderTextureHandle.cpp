#include "../../../../include/graphics/platform/directx/DirectXRenderTextureHandle.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"
#include "../../../../include/core/Log.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

using namespace PhysicsEngine;

static DXGI_FORMAT getTextureFormat(TextureFormat format)
{
    DXGI_FORMAT directxFormat = DXGI_FORMAT_R32_FLOAT;

    switch (format)
    {
    case TextureFormat::Depth:
        directxFormat = DXGI_FORMAT_R32_FLOAT;
        break;
    case TextureFormat::RG:
        directxFormat = DXGI_FORMAT_R8G8_UNORM;
        break;
    case TextureFormat::RGB:
        directxFormat = DXGI_FORMAT_R32G32B32_UINT;
        break;
    case TextureFormat::RGBA:
        directxFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
        break;
    default:
        Log::error("DirectX: Invalid texture format\n");
        break;
    }

    return directxFormat;
}

static D3D11_TEXTURE_ADDRESS_MODE getTextureWrapMode(TextureWrapMode wrapMode)
{
    D3D11_TEXTURE_ADDRESS_MODE directxWrapMode = D3D11_TEXTURE_ADDRESS_WRAP;

    switch (wrapMode)
    {
    case TextureWrapMode::Repeat:
        directxWrapMode = D3D11_TEXTURE_ADDRESS_WRAP;
        break;
    case TextureWrapMode::ClampToEdge:
        directxWrapMode = D3D11_TEXTURE_ADDRESS_CLAMP;
        break;
    case TextureWrapMode::ClampToBorder:
        directxWrapMode = D3D11_TEXTURE_ADDRESS_BORDER;
        break;
    case TextureWrapMode::MirrorRepeat:
        directxWrapMode = D3D11_TEXTURE_ADDRESS_MIRROR;
        break;
    case TextureWrapMode::MirrorClampToEdge:
        directxWrapMode = D3D11_TEXTURE_ADDRESS_MIRROR_ONCE;
        break;
    default:
        Log::error("DirectX: Invalid texture wrap mode\n");
        break;
    }

    return directxWrapMode;
}

DirectXRenderTextureHandle::DirectXRenderTextureHandle(int width, int height, TextureFormat format,
                                                       TextureWrapMode wrapMode, TextureFilterMode filterMode)
    : RenderTextureHandle(width, height, format, wrapMode, filterMode)
{
    mTexture = nullptr;
    mShaderResourceView = nullptr;
    mSamplerState = nullptr;

     mFormat = format;
    mWrapMode = wrapMode;
    mFilterMode = filterMode;
    mAnisoLevel = 1;
    mWidth = width;
    mHeight = height;

    if (width == 0 || height == 0)
    {
        return;
    }

    if (mTexture != nullptr)
    {
        mTexture->Release();
    }
    if (mShaderResourceView != nullptr)
    {
        mShaderResourceView->Release();
    }
    if (mSamplerState != nullptr)
    {
        mSamplerState->Release();
    }

    ZeroMemory(&mTextureDesc, sizeof(D3D11_TEXTURE2D_DESC));

    mTextureDesc.Width = width;
    mTextureDesc.Height = height;
    mTextureDesc.MipLevels = 1;
    mTextureDesc.ArraySize = 1;
    mTextureDesc.Format = getTextureFormat(format);
    mTextureDesc.SampleDesc.Count = 1;
    mTextureDesc.Usage = D3D11_USAGE_DYNAMIC;
    /*mTextureDesc.Usage = D3D11_USAGE_DEFAULT;
    switch (format)
    {
    case TextureFormat::Depth:
        mTextureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL | D3D11_BIND_SHADER_RESOURCE;
        break;
    case TextureFormat::RG:
    case TextureFormat::RGB:
    case TextureFormat::RGBA:
        mTextureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        break;
    }*/
    mTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    mTextureDesc.CPUAccessFlags = 0;
    mTextureDesc.MiscFlags = 0;

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    CHECK_ERROR(device->CreateTexture2D(&mTextureDesc, NULL, &mTexture));

    ZeroMemory(&mShaderResourceViewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
    mShaderResourceViewDesc.Format = mTextureDesc.Format;
    mShaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    mShaderResourceViewDesc.Texture2D.MipLevels = mTextureDesc.MipLevels;

    CHECK_ERROR(device->CreateShaderResourceView(mTexture, &mShaderResourceViewDesc, &mShaderResourceView));

    /*ZeroMemory(&mSamplerDesc, sizeof(D3D11_SAMPLER_DESC));
    mSamplerDesc.AddressU = getTextureWrapMode(wrapMode);
    mSamplerDesc.AddressV = getTextureWrapMode(wrapMode);
    mSamplerDesc.MinLOD = 0;
    mSamplerDesc.MaxLOD = 11;
    mSamplerDesc.Filter =
        filterMode == TextureFilterMode::Bilinear ? D3D11_FILTER_MIN_MAG_MIP_LINEAR : D3D11_FILTER_MIN_MAG_MIP_POINT;
    mSamplerDesc.ComparisonFunc = D3D11_COMPARISON_NEVER;
    mSamplerDesc.MaxLOD = D3D11_FLOAT32_MAX;

    CHECK_ERROR(device->CreateSamplerState(&mSamplerDesc, &mSamplerState));*/
}

DirectXRenderTextureHandle::~DirectXRenderTextureHandle()
{
    if (mTexture != nullptr)
    {
        mTexture->Release();
    }
    if (mShaderResourceView != nullptr)
    {
        mShaderResourceView->Release();
    }
    if (mSamplerState != nullptr)
    {
        mSamplerState->Release();
    }
}

void *DirectXRenderTextureHandle::getTexture()
{
    return static_cast<void*>(mTexture);
}

void *DirectXRenderTextureHandle::getIMGUITexture()
{
    return static_cast<void*>(mShaderResourceView);
}
//
//
////namespace Graphics
////{
////	ShaderProgram
////  Texture2D
////  Cubemap
////  Framebuffer
////  VertexBuffer
////  UniformBuffer
////  Mesh
////}
////
////
////Graphics::Texture2D
////Graphics::ShaderProgram