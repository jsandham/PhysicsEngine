#include "../../../../include/graphics/platform/directx/DirectXShaderProgram.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <algorithm>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")

using namespace PhysicsEngine;

DirectXShaderProgram::DirectXShaderProgram()
{
    mVertexShader = NULL;
    mPixelShader = NULL;
    mGeometryShader = NULL;

    mVertexShaderBlob = NULL;
    mPixelShaderBlob = NULL;
    mGeometryShaderBlob = NULL;


    ZeroMemory(&mVSConstantBufferDesc, sizeof(D3D11_BUFFER_DESC));
    mVSConstantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;   // write access access by CPU and GPU
    mVSConstantBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER; // use as a vertex buffer
    mVSConstantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE; // allow CPU to write in buffer
}

DirectXShaderProgram ::~DirectXShaderProgram()
{
    if (mVertexShader != NULL)
    {
        mVertexShader->Release();
    }
    if (mPixelShader != NULL)
    {
        mPixelShader->Release();
    }
    if (mGeometryShader != NULL)
    {
        mGeometryShader->Release();
    }

    if (mVertexShaderBlob != NULL)
    {
        mVertexShaderBlob->Release();
    }
    if (mPixelShaderBlob != NULL)
    {
        mPixelShaderBlob->Release();
    }
    if (mGeometryShaderBlob != NULL)
    {
        mGeometryShaderBlob->Release();
    }
}

void DirectXShaderProgram::load(const std::string& name, const std::string &vertex, const std::string &fragment, const std::string &geometry)
{
    mName = name;
    mVertex = vertex;
    mFragment = fragment;
    mGeometry = geometry;
}

void DirectXShaderProgram::load(const std::string& name, const std::string &vertex, const std::string &fragment)
{
    this->load(name, vertex, fragment, "");
}

void DirectXShaderProgram::compile()
{
    memset(mStatus.mVertexCompileLog, 0, sizeof(mStatus.mVertexCompileLog));
    memset(mStatus.mFragmentCompileLog, 0, sizeof(mStatus.mFragmentCompileLog));
    memset(mStatus.mGeometryCompileLog, 0, sizeof(mStatus.mGeometryCompileLog));
    memset(mStatus.mLinkLog, 0, sizeof(mStatus.mLinkLog));

    mStatus.mVertexShaderCompiled = 1;
    mStatus.mFragmentShaderCompiled = 1;
    mStatus.mGeometryShaderCompiled = 1;

    if (mVertexShader != NULL)
    {
        mVertexShader->Release();
    }
    if (mPixelShader != NULL)
    {
        mPixelShader->Release();
    }
    if (mGeometryShader != NULL)
    {
        mGeometryShader->Release();
    }

    if (mVertexShaderBlob != NULL)
    {
        mVertexShaderBlob->Release();
    }
    if (mPixelShaderBlob != NULL)
    {
        mPixelShaderBlob->Release();
    }
    if (mGeometryShaderBlob != NULL)
    {
        mGeometryShaderBlob->Release();
    }

    UINT flags = D3DCOMPILE_WARNINGS_ARE_ERRORS | D3DCOMPILE_DEBUG;

    ID3DBlob *errorBlob = nullptr;

    HRESULT result;

    // Compile vertex shader shader
    result = D3DCompile(mVertex.data(), mVertex.size(), NULL, NULL, NULL, "VSMain", "vs_5_0", flags, 0,
                    &mVertexShaderBlob, &errorBlob);
    
    if (FAILED(result))
    {
        std::string message = "Shader: Vertex shader compilation failed (" + mName + ")\n";
        Log::error(message.c_str());

        mStatus.mVertexShaderCompiled = 0;
        if (errorBlob)
        {
            memcpy(mStatus.mVertexCompileLog, errorBlob->GetBufferPointer(), 
                   std::min((size_t)512, (size_t)errorBlob->GetBufferSize()));
            Log::error((char*)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
    }

    // Compile pixel shader shader
    result = D3DCompile(mFragment.data(), mFragment.size(), NULL, NULL, NULL, "PSMain", "ps_5_0", flags,
                    0, &mPixelShaderBlob, &errorBlob);
    if (FAILED(result))
    {
        std::string message = "Shader: Pixel shader compilation failed (" + mName + ")\n";
        Log::error(message.c_str());

        mStatus.mFragmentShaderCompiled = 0;
        if (errorBlob)
        {
            memcpy(mStatus.mFragmentCompileLog, errorBlob->GetBufferPointer(),
                   std::min((size_t)512, (size_t)errorBlob->GetBufferSize()));
            Log::error((char *)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
    }

    if (!mGeometry.empty())
    {
        // Compile pixel shader shader
        result = D3DCompile(mGeometry.data(), mGeometry.size(), NULL, NULL, NULL, "GSMain", "ps_5_0", flags, 0,
                        &mGeometryShaderBlob, &errorBlob);
        if (FAILED(result))
        {
            std::string message = "Shader: Geometry shader compilation failed (" + mName + ")\n";
            Log::error(message.c_str());

            mStatus.mGeometryShaderCompiled = 0;
            if (errorBlob)
            {
                memcpy(mStatus.mGeometryCompileLog, errorBlob->GetBufferPointer(),
                       std::min((size_t)512, (size_t)errorBlob->GetBufferSize()));
                Log::error((char *)errorBlob->GetBufferPointer());
                errorBlob->Release();
            }
        }
    }

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    if (mVertexShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreateVertexShader(
            mVertexShaderBlob->GetBufferPointer(), mVertexShaderBlob->GetBufferSize(), NULL, &mVertexShader));
    }

    if (mPixelShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreatePixelShader(
            mPixelShaderBlob->GetBufferPointer(), mPixelShaderBlob->GetBufferSize(), NULL, &mPixelShader));
    }

    if (mGeometryShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreateGeometryShader(
            mGeometryShaderBlob->GetBufferPointer(), mGeometryShaderBlob->GetBufferSize(), NULL, &mGeometryShader));
    }
}

void DirectXShaderProgram::bind()
{
    DirectXRenderContext::get()->getD3DDeviceContext()->VSSetShader(mVertexShader, NULL, 0);
    DirectXRenderContext::get()->getD3DDeviceContext()->PSSetShader(mPixelShader, NULL, 0);

    //DirectXRenderContext::get()->getD3DDeviceContext()->VSSetConstantBuffers(0, mVSConstantBuffersCount,
    //                                                                         mVSConstantBuffers);
    //DirectXRenderContext::get()->getD3DDeviceContext()->PSSetConstantBuffers(0, mPSConstantBuffersCount,
    //                                                                         mPSConstantBuffers);
}

void DirectXShaderProgram::unbind()
{
    DirectXRenderContext::get()->getD3DDeviceContext()->VSSetShader(NULL, NULL, 0);
    DirectXRenderContext::get()->getD3DDeviceContext()->PSSetShader(NULL, NULL, 0);
}

int DirectXShaderProgram::findUniformLocation(const std::string &name) const
{
    return -1;
}

std::vector<ShaderUniform> DirectXShaderProgram::getUniforms() const
{
    return std::vector<ShaderUniform>();
}

std::vector<ShaderAttribute> DirectXShaderProgram::getAttributes() const
{
    return std::vector<ShaderAttribute>();
}

void DirectXShaderProgram::setBool(const char *name, bool value)
{

}

void DirectXShaderProgram::setInt(const char *name, int value)
{

}

void DirectXShaderProgram::setFloat(const char *name, float value)
{

}

void DirectXShaderProgram::setColor(const char *name, const Color &color)
{

}

void DirectXShaderProgram::setColor32(const char *name, const Color32 &color)
{

}

void DirectXShaderProgram::setVec2(const char *name, const glm::vec2 &vec)
{

}

void DirectXShaderProgram::setVec3(const char *name, const glm::vec3 &vec)
{

}

void DirectXShaderProgram::setVec4(const char *name, const glm::vec4 &vec)
{

}

void DirectXShaderProgram::setMat2(const char *name, const glm::mat2 &mat)
{

}

void DirectXShaderProgram::setMat3(const char *name, const glm::mat3 &mat)
{

}

void DirectXShaderProgram::setMat4(const char *name, const glm::mat4 &mat)
{

}

void DirectXShaderProgram::setTexture2D(const char *name, int texUnit, TextureHandle* tex)
{

}

void DirectXShaderProgram::setTexture2Ds(const char *name, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs)
{

}

void DirectXShaderProgram::setBool(int nameLocation, bool value)
{
}

void DirectXShaderProgram::setInt(int nameLocation, int value)
{
}

void DirectXShaderProgram::setFloat(int nameLocation, float value)
{
}

void DirectXShaderProgram::setColor(int nameLocation, const Color &color)
{
}

void DirectXShaderProgram::setColor32(int nameLocation, const Color32 &color)
{
}

void DirectXShaderProgram::setVec2(int nameLocation, const glm::vec2 &vec)
{
}

void DirectXShaderProgram::setVec3(int nameLocation, const glm::vec3 &vec)
{
}

void DirectXShaderProgram::setVec4(int nameLocation, const glm::vec4 &vec)
{
}

void DirectXShaderProgram::setMat2(int nameLocation, const glm::mat2 &mat)
{
}

void DirectXShaderProgram::setMat3(int nameLocation, const glm::mat3 &mat)
{
}

void DirectXShaderProgram::setMat4(int nameLocation, const glm::mat4 &mat)
{
}

void DirectXShaderProgram::setTexture2D(int nameLocation, int texUnit, TextureHandle* tex)
{
}

void DirectXShaderProgram::setTexture2Ds(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs)
{
}

bool DirectXShaderProgram::getBool(const char *name) const
{
    return false;
}

int DirectXShaderProgram::getInt(const char *name) const
{
    return -1;
}

float DirectXShaderProgram::getFloat(const char *name) const
{
    return 0.0f;
}

Color DirectXShaderProgram::getColor(const char *name) const
{
    return Color::black;
}

glm::vec2 DirectXShaderProgram::getVec2(const char *name) const
{
    return glm::vec2();
}

glm::vec3 DirectXShaderProgram::getVec3(const char *name) const
{
    return glm::vec3();
}

glm::vec4 DirectXShaderProgram::getVec4(const char *name) const
{
    return glm::vec4();
}

glm::mat2 DirectXShaderProgram::getMat2(const char *name) const
{
    return glm::mat2();
}

glm::mat3 DirectXShaderProgram::getMat3(const char *name) const
{
    return glm::mat3();
}

glm::mat4 DirectXShaderProgram::getMat4(const char *name) const
{
    return glm::mat4();
}

bool DirectXShaderProgram::getBool(int nameLocation) const 
{
    return false;
}

int DirectXShaderProgram::getInt(int nameLocation) const 
{
    return -1;
}

float DirectXShaderProgram::getFloat(int nameLocation) const 
{
    return 0.0f;
}

Color DirectXShaderProgram::getColor(int nameLocation) const 
{
    return Color::black;
}

Color32 DirectXShaderProgram::getColor32(int nameLocation) const 
{
    return Color32::black;
}

glm::vec2 DirectXShaderProgram::getVec2(int nameLocation) const
{
    return glm::vec2();
}

glm::vec3 DirectXShaderProgram::getVec3(int nameLocation) const
{
    return glm::vec3();
}

glm::vec4 DirectXShaderProgram::getVec4(int nameLocation) const
{
    return glm::vec4();
}

glm::mat2 DirectXShaderProgram::getMat2(int nameLocation) const
{
    return glm::mat2();
}

glm::mat3 DirectXShaderProgram::getMat3(int nameLocation) const
{
    return glm::mat3();
}

glm::mat4 DirectXShaderProgram::getMat4(int nameLocation) const
{
    return glm::mat4();
}