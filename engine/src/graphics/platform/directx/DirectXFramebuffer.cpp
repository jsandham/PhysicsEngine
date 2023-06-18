#include "../../../../include/core/Log.h"
#include "../../../../include/graphics/platform/directx/DirectXFramebuffer.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"
#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <glm/glm.hpp>

using namespace PhysicsEngine;

DirectXFramebuffer::DirectXFramebuffer(int width, int height) : Framebuffer(width, height)
{
    mColorTex.resize(1);
    mRenderTargetViews.resize(1);
    mNullRenderTargetViews.resize(1);

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
   
    mColorTex[0] = RenderTextureHandle::create(mWidth, mHeight, TextureFormat::RGBA, TextureWrapMode::ClampToEdge,
                                         TextureFilterMode::Nearest);
    mDepthTex = RenderTextureHandle::create(mWidth, mHeight, TextureFormat::Depth, TextureWrapMode::ClampToEdge,
                                      TextureFilterMode::Nearest);

    // Creating a view of the texture to be used when binding it as a render target
    // D3D11_RENDER_TARGET_VIEW_DESC renderTargetViewDesc;
    // renderTargetViewDesc.Format = format;
    // renderTargetViewDesc.ViewDimension = D3D11_RTV_DIMENSION_TEXTURE2D;
    // renderTargetViewDesc.Texture2D.MipSlice = 0;
    CHECK_ERROR(device->CreateRenderTargetView(static_cast<ID3D11Texture2D*>(mColorTex[0]->getTexture()), nullptr,
                                               &mRenderTargetViews[0]));
    CHECK_ERROR(device->CreateDepthStencilView(static_cast<ID3D11Texture2D*>(mDepthTex->getTexture()), nullptr,
                                               &mDepthStencilView));


    //// D3D Objects To Create Into
    //ID3D11Texture2D *_Texture2D = NULL;
    //ID3D11RenderTargetView *_RenderTargetView = NULL;
    //ID3D11ShaderResourceView *_ShaderResourceView = NULL;

    //// D3D Device
    //ID3D11Device *Device = _directX->GetDevice();

    //D3D11_TEXTURE2D_DESC bufferDesc;
    //bufferDesc.ArraySize = 1;
    //bufferDesc.BindFlags = D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE;
    //bufferDesc.CPUAccessFlags = 0;
    //bufferDesc.Format = format;
    //bufferDesc.Height = height;
    //bufferDesc.MipLevels = 1;
    //bufferDesc.MiscFlags = 0;
    //bufferDesc.SampleDesc = sampleDesc;
    //bufferDesc.Usage = D3D11_USAGE_DEFAULT;
    //bufferDesc.Width = width;
    //HRESULT hr = Device->CreateTexture2D(&bufferDesc, 0, &_Texture2D);

    //// Creating a view of the texture to be used when binding it on a shader to sample
    //D3D11_SHADER_RESOURCE_VIEW_DESC shaderResourceViewDesc;
    //shaderResourceViewDesc.Format = format;
    //shaderResourceViewDesc.ViewDimension = D3D11_SRV_DIMENSION_TEXTURE2D;
    //shaderResourceViewDesc.Texture2D.MostDetailedMip = 0;
    //shaderResourceViewDesc.Texture2D.MipLevels = 1;
    //hr = Device->CreateShaderResourceView(_Texture2D, &shaderResourceViewDesc, &_ShaderResourceView);
}

DirectXFramebuffer::DirectXFramebuffer(int width, int height, int numColorTex, bool addDepthTex)
    : Framebuffer(width, height, numColorTex, addDepthTex)
{
    mColorTex.resize(mNumColorTex);
    mRenderTargetViews.resize(mNumColorTex);
    mNullRenderTargetViews.resize(mNumColorTex);

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();

    for (size_t i = 0; i < mColorTex.size(); i++)
    {
        mColorTex[i] = RenderTextureHandle::create(mWidth, mHeight, TextureFormat::RGBA, TextureWrapMode::ClampToEdge,
                                             TextureFilterMode::Nearest);
        CHECK_ERROR(device->CreateRenderTargetView(static_cast<ID3D11Texture2D*>(mColorTex[i]->getTexture()), nullptr,
                                                   &mRenderTargetViews[i]));
    }

    if (mAddDepthTex)
    {
        mDepthTex = RenderTextureHandle::create(mWidth, mHeight, TextureFormat::Depth, TextureWrapMode::ClampToEdge,
                                          TextureFilterMode::Nearest);
        CHECK_ERROR(device->CreateDepthStencilView(static_cast<ID3D11Texture2D*>(mDepthTex->getTexture()), nullptr,
                                                   &mDepthStencilView));
    }
    else
    {
        mDepthTex = nullptr;
    }
}

DirectXFramebuffer::~DirectXFramebuffer()
{
    // delete textures
    for (size_t i = 0; i < mColorTex.size(); i++)
    {
        if (mRenderTargetViews[i] != nullptr)
        {
            mRenderTargetViews[i]->Release();
        }
        delete mColorTex[i];
    }

    if (mAddDepthTex)
    {
        if (mDepthStencilView != nullptr)
        {
            mDepthStencilView->Release();
        }
        delete mDepthTex;
    }
}

void DirectXFramebuffer::clearColor(Color color)
{
    this->clearColor(color.mR, color.mG, color.mB, color.mA);
}

void DirectXFramebuffer::clearColor(float r, float g, float b, float a)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    float color[4];
    color[0] = r;
    color[1] = g;
    color[2] = b;
    color[3] = a;

    context->ClearRenderTargetView(mRenderTargetViews[0], color);
}

void DirectXFramebuffer::clearDepth(float depth)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    context->ClearDepthStencilView(mDepthStencilView, D3D11_CLEAR_DEPTH, depth, 0);
}

void DirectXFramebuffer::bind()
{
    //ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    //assert(context != nullptr);

    //context->OMSetRenderTargets(mRenderTargetViews.size(), mRenderTargetViews.data(), mDepthStencilView);
}

void DirectXFramebuffer::unbind()
{
    //ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    //assert(context != nullptr);

    //context->OMSetRenderTargets(mNullRenderTargetViews.size(), mNullRenderTargetViews.data(), nullptr);
}

void DirectXFramebuffer::setViewport(int x, int y, int width, int height)
{
    assert(x >= 0);
    assert(y >= 0);
    assert((unsigned int)(x + width) <= mWidth);
    assert((unsigned int)(y + height) <= mHeight);

    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    D3D11_VIEWPORT viewport;
    viewport.Width = static_cast<float>(mWidth);
    viewport.Height = static_cast<float>(mHeight);
    viewport.TopLeftX = static_cast<float>(x);
    viewport.TopLeftY = static_cast<float>(y);

    context->RSSetViewports(1, &viewport);
}

RenderTextureHandle *DirectXFramebuffer::getColorTex(size_t i)
{
    assert(i < mColorTex.size());
    return mColorTex[i];
}

RenderTextureHandle *DirectXFramebuffer::getDepthTex()
{
    return mDepthTex;
}