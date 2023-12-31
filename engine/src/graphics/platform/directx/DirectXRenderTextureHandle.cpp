#include "../../../../include/graphics/platform/directx/DirectXRenderTextureHandle.h"
#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

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

    // Setup the render target texture description.
    ZeroMemory(&mTextureDesc, sizeof(D3D11_TEXTURE2D_DESC));
    mTextureDesc.Width = width;
    mTextureDesc.Height = height;
    mTextureDesc.MipLevels = 1;
    mTextureDesc.ArraySize = 1;
    mTextureDesc.SampleDesc.Count = 1;
    mTextureDesc.Usage = D3D11_USAGE_DEFAULT;
    switch (format)
    {
    case TextureFormat::Depth:
        mTextureDesc.Format = DXGI_FORMAT_D32_FLOAT;
        mTextureDesc.BindFlags = D3D11_BIND_DEPTH_STENCIL;
        break;
    case TextureFormat::RG:
        mTextureDesc.Format = DXGI_FORMAT_R32G32_FLOAT;
        mTextureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    case TextureFormat::RGB:
        mTextureDesc.Format = DXGI_FORMAT_R32G32B32_FLOAT;
        mTextureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        break;
    case TextureFormat::RGBA:
        mTextureDesc.Format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        mTextureDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
        break;
    }
    mTextureDesc.CPUAccessFlags = 0;
    mTextureDesc.MiscFlags = 0;

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    CHECK_ERROR(device->CreateTexture2D(&mTextureDesc, NULL, &mTexture));

    switch (format)
    {
    case TextureFormat::Depth:
        break;
    case TextureFormat::RG:
    case TextureFormat::RGB:
    case TextureFormat::RGBA:
        ZeroMemory(&mShaderResourceViewDesc, sizeof(D3D11_SHADER_RESOURCE_VIEW_DESC));
        mShaderResourceViewDesc.Format = mTextureDesc.Format;
        mShaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
        mShaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
        mShaderResourceViewDesc.Texture2D.MipLevels = mTextureDesc.MipLevels;

        CHECK_ERROR(device->CreateShaderResourceView(mTexture, &mShaderResourceViewDesc, &mShaderResourceView));
        break;
    }
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

void DirectXRenderTextureHandle::load(const std::vector<unsigned char> &data)
{

}

void *DirectXRenderTextureHandle::getTexture()
{
    return static_cast<void *>(mTexture);
}

void *DirectXRenderTextureHandle::getIMGUITexture()
{
    return static_cast<void *>(mShaderResourceView);
}