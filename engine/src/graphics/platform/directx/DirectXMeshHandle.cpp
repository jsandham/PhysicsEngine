#include "../../../../include/graphics/platform/directx/DirectXMeshHandle.h"
#include "../../../../include/graphics/platform/directx/DirectXShaderProgram.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <assert.h>

using namespace PhysicsEngine;

struct ShaderBlobHelper
{
    static ID3DBlob *createShaderBlobFromInputLayout(const std::vector<D3D11_INPUT_ELEMENT_DESC> &descr);
};

ID3DBlob *ShaderBlobHelper::createShaderBlobFromInputLayout(const std::vector<D3D11_INPUT_ELEMENT_DESC> &descr)
{
    // Create shader string
    std::string shaderStr;
    
    std::string s = "struct VS_INPUT{\n";
    std::string e = "};\n";
    std::string ep = "VS_INPUT VSMain(VS_INPUT input)\n"
                     "{\n"
                     "  return input;\n"
                     "}\n";

    shaderStr += s;

    size_t i = 0;
    while (i < descr.size())
    {
        std::string format;
        switch (descr[i].Format)
        {
        case DXGI_FORMAT_R32_SINT: {
            format = "int ";
            break;
        }
        case DXGI_FORMAT_R32_FLOAT: {
            format = "float ";
            break;
        }
        case DXGI_FORMAT_R32G32_FLOAT: {
            format = "float2 ";
            break;
        }
        case DXGI_FORMAT_R32G32B32_FLOAT: {
            format = "float3 ";
            break;
        }
        case DXGI_FORMAT_R32G32B32A32_FLOAT: {
            format = "float4 ";
            break;
        }
        case DXGI_FORMAT_R32G32_SINT: {
            format = "int2 ";
            break;
        }
        case DXGI_FORMAT_R32G32B32_SINT: {
            format = "int3 ";
            break;
        }
        case DXGI_FORMAT_R32G32B32A32_SINT: {
            format = "int4 ";
            break;
        }
        case DXGI_FORMAT_R32G32_UINT: {
            format = "uint2 ";
            break;
        }
        case DXGI_FORMAT_R32G32B32_UINT: {
            format = "uint3 ";
            break;
        }
        case DXGI_FORMAT_R32G32B32A32_UINT: {
            format = "uint4 ";
            break;
        }
        }

        if (descr[i].AlignedByteOffset == D3D11_APPEND_ALIGNED_ELEMENT)
        {
            format = "matrix ";
            shaderStr += format + descr[i].SemanticName + " : " + descr[i].SemanticName + ";\n";
            i += 4;
        }
        else
        {
            shaderStr += format + descr[i].SemanticName + " : " + descr[i].SemanticName + ";\n";
            i++;
        }
    }
    shaderStr += e;
    shaderStr += ep;

    UINT flags = D3DCOMPILE_WARNINGS_ARE_ERRORS | D3DCOMPILE_DEBUG;
    ID3DBlob *shaderBlob;
    ID3DBlob *errorBlob = nullptr;
    HRESULT result;

    // Compile vertex shader shader
    result = D3DCompile(shaderStr.data(), shaderStr.size(), NULL, NULL, NULL, "VSMain", "vs_5_0", flags, 0,
                        &shaderBlob, &errorBlob);

    if (FAILED(result))
    {
        std::string message = "createShaderBlobFromInputLayout failed\n";
        Log::error(message.c_str());

        if (errorBlob)
        {
            std::string test = std::string((char*)errorBlob->GetBufferPointer());
            Log::error((char *)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }

        return NULL;
    }

    return shaderBlob;
}

DirectXMeshHandle::DirectXMeshHandle()
{
    mIndexBuffer = NULL;
    mBufferLayout = NULL;
}

DirectXMeshHandle::~DirectXMeshHandle()
{
    mBufferLayout->Release();
}

void DirectXMeshHandle::addVertexBuffer(VertexBuffer *buffer, std::string name, AttribType type, bool instanceBuffer)
{
    assert(buffer != nullptr);

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    assert(mInputDescs.size() == mInputSemanticNames.size());

    int increment = (type == AttribType::Mat4) ? 4 : 1;

    UINT layoutSlot = (UINT)mInputDescs.size();
    mInputDescs.resize(layoutSlot + increment);
    mInputSemanticNames.resize(layoutSlot + increment);

    assert(mInputDescs.size() == mInputSemanticNames.size());

    // Add description of layout to our cached layout list
    DXGI_FORMAT format = DXGI_FORMAT_UNKNOWN; 
    switch (type)
    {
    case AttribType::Int: {
        format = DXGI_FORMAT_R32_SINT;
        break;
    }
    case AttribType::Float: {
        format = DXGI_FORMAT_R32_FLOAT;
        break;
    }
    case AttribType::Vec2: {
        format = DXGI_FORMAT_R32G32_FLOAT;
        break;
    }
    case AttribType::Vec3: {
        format = DXGI_FORMAT_R32G32B32_FLOAT;
        break;
    }
    case AttribType::Vec4: {
        format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        break;
    }
    case AttribType::IVec2: {
        format = DXGI_FORMAT_R32G32_SINT;
        break;
    }
    case AttribType::IVec3: {
        format = DXGI_FORMAT_R32G32B32_SINT;
        break;
    }
    case AttribType::IVec4: {
        format = DXGI_FORMAT_R32G32B32A32_SINT;
        break;
    }
    case AttribType::UVec2: {
        format = DXGI_FORMAT_R32G32_UINT;
        break;
    }
    case AttribType::UVec3: {
        format = DXGI_FORMAT_R32G32B32_UINT;
        break;
    }
    case AttribType::UVec4: {
        format = DXGI_FORMAT_R32G32B32A32_UINT;
        break;
    }
    case AttribType::Mat4: {
        format = DXGI_FORMAT_R32G32B32A32_FLOAT;
        break;
    }
    }

    D3D11_INPUT_CLASSIFICATION classification =
        instanceBuffer ? D3D11_INPUT_PER_INSTANCE_DATA : D3D11_INPUT_PER_VERTEX_DATA;
    UINT instanceStepRate = instanceBuffer ? 1 : 0;

    if (type == AttribType::Mat4)
    {
        mInputSemanticNames[layoutSlot + 0] = name;
        mInputSemanticNames[layoutSlot + 1] = name;
        mInputSemanticNames[layoutSlot + 2] = name;
        mInputSemanticNames[layoutSlot + 3] = name;

        mInputDescs[layoutSlot + 0] = {
            "", 0, format, layoutSlot, D3D11_APPEND_ALIGNED_ELEMENT, classification, instanceStepRate};
        mInputDescs[layoutSlot + 1] = {
            "", 1, format, layoutSlot, D3D11_APPEND_ALIGNED_ELEMENT, classification, instanceStepRate};
        mInputDescs[layoutSlot + 2] = {
            "", 2, format, layoutSlot, D3D11_APPEND_ALIGNED_ELEMENT, classification, instanceStepRate};
        mInputDescs[layoutSlot + 3] = {
            "", 3, format, layoutSlot, D3D11_APPEND_ALIGNED_ELEMENT, classification, instanceStepRate};
    }
    else
    {
        mInputSemanticNames[layoutSlot] = name;
        mInputDescs[layoutSlot] = {"", 0, format, layoutSlot, 0, classification, instanceStepRate};
    }

    if (mBufferLayout != NULL)
    {
        mBufferLayout->Release();
        mBufferLayout = NULL;
    }

    assert(mBufferLayout == NULL);

    for (size_t i = 0; i < mInputDescs.size(); i++)
    {
        mInputDescs[i].SemanticName = mInputSemanticNames[i].c_str();
    }

    ID3DBlob *blob = ShaderBlobHelper::createShaderBlobFromInputLayout(mInputDescs); // this is so annoying DirectX!
    CHECK_ERROR(device->CreateInputLayout(mInputDescs.data(), (UINT)mInputDescs.size(), blob->GetBufferPointer(),
                                          blob->GetBufferSize(), &mBufferLayout));
    
    mBuffers.push_back(buffer);

    blob->Release();
}

void DirectXMeshHandle::addIndexBuffer(IndexBuffer *buffer)
{
    assert(buffer != nullptr);

    mIndexBuffer = buffer;
}

void DirectXMeshHandle::bind()
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
    context->IASetInputLayout(mBufferLayout);

    for (unsigned int i = 0; i < 2 /*mBuffers.size()*/; i++)
    {
        mBuffers[i]->bind(i);
    }

    if (mIndexBuffer != nullptr)
    {
        mIndexBuffer->bind();
    }
}

void DirectXMeshHandle::unbind()
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    context->IASetPrimitiveTopology(D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED);
    context->IASetInputLayout(NULL);

    for (unsigned int i = 0; i < 2/*mBuffers.size()*/; i++)
    {
        mBuffers[i]->unbind(i);
    }

    if (mIndexBuffer != nullptr)
    {
        mIndexBuffer->unbind();
    }
}

void DirectXMeshHandle::drawLines(size_t vertexOffset, size_t vertexCount)
{
}

void DirectXMeshHandle::draw(size_t vertexOffset, size_t vertexCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    this->bind();
    context->Draw((unsigned int)vertexCount, (unsigned int)vertexOffset);
    this->unbind();
}

void DirectXMeshHandle::drawIndexed(size_t indexOffset, size_t indexCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    this->bind();
    context->DrawIndexed((unsigned int)indexCount, (unsigned int)indexOffset, 0);
    this->unbind();
}

void DirectXMeshHandle::drawInstanced(size_t vertexOffset, size_t vertexCount, size_t instanceCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    this->bind();
    context->DrawInstanced((unsigned int)vertexCount, (unsigned int)instanceCount, (unsigned int)vertexOffset, 0);
    this->unbind();
}

void DirectXMeshHandle::drawIndexedInstanced(size_t indexOffset, size_t indexCount, size_t instanceCount)
{
    ID3D11DeviceContext *context = DirectXRenderContext::get()->getD3DDeviceContext();
    assert(context != nullptr);

    this->bind();
    context->DrawIndexedInstanced((unsigned int)indexCount, (unsigned int)instanceCount, (unsigned int)indexOffset,
                                      0, 0);
    this->unbind();
}