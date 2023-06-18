#include "../../../../include/graphics/platform/directx/DirectXTextureHandle.h"
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

DirectXTextureHandle::DirectXTextureHandle(int width, int height, TextureFormat format, TextureWrapMode wrapMode,
                                           TextureFilterMode filterMode)
    : TextureHandle(width, height, format, wrapMode, filterMode)
{
    mTexture = nullptr;
    mShaderResourceView = nullptr;
    mSamplerState = nullptr;

    this->load(format, wrapMode, filterMode, width, height, std::vector<unsigned char>());
}

DirectXTextureHandle::~DirectXTextureHandle()
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

void DirectXTextureHandle::load(TextureFormat format,
	TextureWrapMode wrapMode,
	TextureFilterMode filterMode,
	int width,
	int height,
	const std::vector<unsigned char>& data)
{
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
    mTextureDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    mTextureDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE;
    mTextureDesc.MiscFlags = 0;

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    if (data.data() != nullptr)
    {
        int numChannels = 1;
        switch (format)
        {
        case TextureFormat::Depth:
            numChannels = 1;
            break;
        case TextureFormat::RG:
            numChannels = 2;
            break;
        case TextureFormat::RGB:
            numChannels = 3;
            break;
        case TextureFormat::RGBA:
            numChannels = 4;
            break;
        }

        D3D11_SUBRESOURCE_DATA initData;
        initData.pSysMem = data.data();
        initData.SysMemPitch = numChannels * sizeof(unsigned char) * mWidth;
        initData.SysMemSlicePitch = 0;

        CHECK_ERROR(device->CreateTexture2D(&mTextureDesc, &initData, &mTexture));
    }
    else
    {
        CHECK_ERROR(device->CreateTexture2D(&mTextureDesc, NULL, &mTexture));
    }

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

void DirectXTextureHandle::update(TextureWrapMode wrapMode, TextureFilterMode filterMode, int anisoLevel)
{

}

void DirectXTextureHandle::readPixels(std::vector<unsigned char>& data)
{

}

void DirectXTextureHandle::writePixels(const std::vector<unsigned char>& data)
{

}

void DirectXTextureHandle::bind(unsigned int texUnit)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();

    assert(context != nullptr);

    context->PSSetShaderResources(texUnit, 1, &mShaderResourceView);
    //context->PSSetSamplers(texUnit, 1, &mSamplerState);
}

void DirectXTextureHandle::unbind(unsigned int texUnit)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();

    assert(context != nullptr);

    ID3D11ShaderResourceView *rv = nullptr;
    context->PSSetShaderResources(texUnit, 1, &rv);

    //ID3D11SamplerState *ss = nullptr;
    //context->PSSetSamplers(texUnit, 1, &ss);
}

void* DirectXTextureHandle::getTexture()
{
    return static_cast<void *>(mTexture);
}

void *DirectXTextureHandle::getIMGUITexture()
{
    return static_cast<void *>(mShaderResourceView);
}