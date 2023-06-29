#include "../../../../include/graphics/platform/directx/DirectXShaderProgram.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include <algorithm>
#include <iostream>
#include <d3dcompiler.h>

#pragma comment(lib, "d3dcompiler.lib")
#pragma comment(lib, "dxguid.lib")

using namespace PhysicsEngine;

DirectXShaderProgram::DirectXShaderProgram()
{
    mVertexShader = NULL;
    mPixelShader = NULL;
    mGeometryShader = NULL;

    mVertexShaderBlob = NULL;
    mPixelShaderBlob = NULL;
    mGeometryShaderBlob = NULL;

    mVertexShaderReflector = NULL;
    mPixelShaderReflector = NULL;
    mGeometryShaderReflector = NULL;

    // This should be replaced with UniformBuffer I think
    //ZeroMemory(&mVSConstantBufferDesc, sizeof(D3D11_BUFFER_DESC));
    //mVSConstantBufferDesc.Usage = D3D11_USAGE_DYNAMIC;   // write access access by CPU and GPU
    //mVSConstantBufferDesc.BindFlags = D3D11_BIND_VERTEX_BUFFER; // use as a vertex buffer
    //mVSConstantBufferDesc.CPUAccessFlags = D3D11_CPU_ACCESS_WRITE; // allow CPU to write in buffer
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

    memset(mStatus.mVertexCompileLog, 0, sizeof(mStatus.mVertexCompileLog));
    memset(mStatus.mFragmentCompileLog, 0, sizeof(mStatus.mFragmentCompileLog));
    memset(mStatus.mGeometryCompileLog, 0, sizeof(mStatus.mGeometryCompileLog));
    memset(mStatus.mLinkLog, 0, sizeof(mStatus.mLinkLog));
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

    if (mVertexShaderReflector != NULL)
    {
        mVertexShaderReflector->Release();
    }
    if (mPixelShaderReflector != NULL)
    {
        mPixelShaderReflector->Release();
    }
    if (mGeometryShaderReflector != NULL)
    {
        mGeometryShaderReflector->Release();
    }

    // Free any existing constant buffers
    for (size_t i = 0; i < mVSConstantBuffers.size(); i++)
    {
        delete mVSConstantBuffers[i];
    }

    for (size_t i = 0; i < mPSConstantBuffers.size(); i++)
    {
        delete mPSConstantBuffers[i];
    }

    for (size_t i = 0; i < mGSConstantBuffers.size(); i++)
    {
        delete mGSConstantBuffers[i];
    }

    mVSConstantBuffers.clear();
    mPSConstantBuffers.clear();
    mGSConstantBuffers.clear();

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
        D3DReflect(mVertexShaderBlob->GetBufferPointer(), mVertexShaderBlob->GetBufferSize(),
                   IID_ID3D11ShaderReflection, (void **)&mVertexShaderReflector);

        D3D11_SHADER_DESC desc;
        CHECK_ERROR(mVertexShaderReflector->GetDesc(&desc));

        std::cout << "mName: " << mName << std::endl;
        std::cout << "Vertex constant buffer count: " << desc.ConstantBuffers << std::endl;

        mVSConstantBuffers.resize(desc.ConstantBuffers);

        // Loop through all constant buffers
        for (unsigned int i = 0; i < desc.ConstantBuffers; i++)
        {
            ID3D11ShaderReflectionConstantBuffer *refectionBuffer =
                mVertexShaderReflector->GetConstantBufferByIndex(i);
            D3D11_SHADER_BUFFER_DESC constantBufferDesc = {};

            CHECK_ERROR(refectionBuffer->GetDesc(&constantBufferDesc));

            std::cout << "constantBufferDesc.Name: " << constantBufferDesc.Name << std::endl;
            std::cout << "constantBufferDesc.Size: " << constantBufferDesc.Size << std::endl;
            std::cout << "constantBufferDesc.Type: " << constantBufferDesc.Type << std::endl;
            std::cout << "constantBufferDesc.uFlags: " << constantBufferDesc.uFlags << std::endl;
            std::cout << "constantBufferDesc.Variables: " << constantBufferDesc.Variables << std::endl;

            D3D11_SHADER_INPUT_BIND_DESC VSInputBindDesc;
            CHECK_ERROR(
                mVertexShaderReflector->GetResourceBindingDescByName(constantBufferDesc.Name, &VSInputBindDesc));

            std::cout << "VSInputBindDesc.BindCount: " << VSInputBindDesc.BindCount << std::endl;
            std::cout << "VSInputBindDesc.BindPoint: " << VSInputBindDesc.BindPoint << std::endl;
            std::cout << "VSInputBindDesc.Dimension: " << VSInputBindDesc.Dimension << std::endl;
            std::cout << "VSInputBindDesc.Name: " << VSInputBindDesc.Name << std::endl;
            std::cout << "VSInputBindDesc.NumSamples: " << VSInputBindDesc.NumSamples << std::endl;
            std::cout << "VSInputBindDesc.ReturnType: " << VSInputBindDesc.ReturnType << std::endl;
            std::cout << "VSInputBindDesc.Type: " << VSInputBindDesc.Type << std::endl;
            std::cout << "VSInputBindDesc.uFlags: " << VSInputBindDesc.uFlags << std::endl;

            mVSConstantBuffers[i] = UniformBuffer::create(constantBufferDesc.Size, VSInputBindDesc.BindPoint);
        }

        /*std::cout << "desc.ArrayInstructionCount: " << desc.ArrayInstructionCount << std::endl;
        std::cout << "desc.BoundResources: " << desc.BoundResources << std::endl;
        std::cout << "desc.cBarrierInstructions: " << desc.cBarrierInstructions << std::endl;
        std::cout << "desc.cControlPoints: " << desc.cControlPoints << std::endl;
        std::cout << "desc.cGSInstanceCount: " << desc.cGSInstanceCount << std::endl;
        std::cout << "desc.cInterlockedInstructions: " << desc.cInterlockedInstructions << std::endl;
        std::cout << "desc.ConstantBuffers: " << desc.ConstantBuffers << std::endl;
        std::cout << "desc.Creator: " << desc.Creator << std::endl;
        std::cout << "desc.cTextureStoreInstructions: " << desc.cTextureStoreInstructions << std::endl;
        std::cout << "desc.CutInstructionCount: " << desc.CutInstructionCount << std::endl;
        std::cout << "desc.DclCount: " << desc.DclCount << std::endl;
        std::cout << "desc.DefCount: " << desc.DefCount << std::endl;
        std::cout << "desc.DynamicFlowControlCount: " << desc.DynamicFlowControlCount << std::endl;
        std::cout << "desc.EmitInstructionCount: " << desc.EmitInstructionCount << std::endl;
        std::cout << "desc.InputParameters: " << desc.InputParameters << std::endl;
        std::cout << "desc.OutputParameters: " << desc.OutputParameters << std::endl;

        ID3D11ShaderReflectionVariable* refectionVariable1 = mVertexShaderReflector->GetVariableByName("worldViewProjection");
        D3D11_SHADER_VARIABLE_DESC desc1;
        refectionVariable1->GetDesc(&desc1);

        std::cout << "mName: " << mName << std::endl;
        std::cout << "desc1.DefaultValue: " << desc1.DefaultValue << std::endl;
        std::cout << "desc1.Name: " << desc1.Name << std::endl;
        std::cout << "desc1.SamplerSize: " << desc1.SamplerSize << std::endl;
        std::cout << "desc1.Size: " << desc1.Size << std::endl;
        std::cout << "desc1.StartOffset: " << desc1.StartOffset << std::endl;
        std::cout << "desc1.StartSampler: " << desc1.StartSampler << std::endl;
        std::cout << "desc1.StartTexture: " << desc1.StartTexture << std::endl;
        std::cout << "desc1.TextureSize: " << desc1.TextureSize << std::endl;
        std::cout << "desc1.uFlags: " << desc1.uFlags << std::endl;


        ID3D11ShaderReflectionConstantBuffer *refectionBuffer1 = mVertexShaderReflector->GetConstantBufferByIndex(0);
        D3D11_SHADER_BUFFER_DESC cbShaderDesc1 = {};

        hr = refectionBuffer1->GetDesc(&cbShaderDesc1);

        if (hr == S_OK)
        {
            std::cout << "cbShaderDesc1.Name: " << cbShaderDesc1.Name << std::endl;
            std::cout << "cbShaderDesc1.Size: " << cbShaderDesc1.Size << std::endl;
            std::cout << "cbShaderDesc1.Type: " << cbShaderDesc1.Type << std::endl;
            std::cout << "cbShaderDesc1.uFlags: " << cbShaderDesc1.uFlags << std::endl;
            std::cout << "cbShaderDesc1.Variables: " << cbShaderDesc1.Variables << std::endl;

        }

        ID3D11ShaderReflectionConstantBuffer *refectionBuffer2 = mVertexShaderReflector->GetConstantBufferByIndex(1);
        D3D11_SHADER_BUFFER_DESC cbShaderDesc2 = {};

        hr = refectionBuffer2->GetDesc(&cbShaderDesc2);

        if (hr == S_OK)
        {
            std::cout << "cbShaderDesc2.Name: " << cbShaderDesc2.Name << std::endl;
            std::cout << "cbShaderDesc2.Size: " << cbShaderDesc2.Size << std::endl;
            std::cout << "cbShaderDesc2.Type: " << cbShaderDesc2.Type << std::endl;
            std::cout << "cbShaderDesc2.uFlags: " << cbShaderDesc2.uFlags << std::endl;
            std::cout << "cbShaderDesc2.Variables: " << cbShaderDesc2.Variables << std::endl;
        }

        ID3D11ShaderReflectionConstantBuffer *refectionBuffer3 =
            mVertexShaderReflector->GetConstantBufferByName("Test2");
        D3D11_SHADER_BUFFER_DESC cbShaderDesc3 = {};

        hr = refectionBuffer3->GetDesc(&cbShaderDesc3);

        if (hr == S_OK)
        {
            std::cout << "cbShaderDesc3.Name: " << cbShaderDesc3.Name << std::endl;
            std::cout << "cbShaderDesc3.Size: " << cbShaderDesc3.Size << std::endl;
            std::cout << "cbShaderDesc3.Type: " << cbShaderDesc3.Type << std::endl;
            std::cout << "cbShaderDesc3.uFlags: " << cbShaderDesc3.uFlags << std::endl;
            std::cout << "cbShaderDesc3.Variables: " << cbShaderDesc3.Variables << std::endl;
        }*/








        /*ID3D11ShaderReflectionVariable* variable2 = mVertexShaderReflector->GetVariableByName("wvp");
        D3D11_SHADER_VARIABLE_DESC desc2;
        HRESULT hr = variable2->GetDesc(&desc2);

        if (hr == S_OK)
        {
            std::cout << "desc2.DefaultValue: " << desc2.DefaultValue << std::endl;
            std::cout << "desc2.Name: " << desc2.Name << std::endl;
            std::cout << "desc2.SamplerSize: " << desc2.SamplerSize << std::endl;
            std::cout << "desc2.Size: " << desc2.Size << std::endl;
            std::cout << "desc2.StartOffset: " << desc2.StartOffset << std::endl;
            std::cout << "desc2.StartSampler: " << desc2.StartSampler << std::endl;
            std::cout << "desc2.StartTexture: " << desc2.StartTexture << std::endl;
            std::cout << "desc2.TextureSize: " << desc2.TextureSize << std::endl;
            std::cout << "desc2.uFlags: " << desc2.uFlags << std::endl;
        }

        hr = mVertexShaderReflector->GetResourceBindingDescByName("Test", &mVSInputBindDesc);

        if (hr == S_OK)
        {
            std::cout << "mVSInputBindDesc.BindCount: " << mVSInputBindDesc.BindCount << std::endl;
            std::cout << "mVSInputBindDesc.BindPoint: " << mVSInputBindDesc.BindPoint << std::endl;
            std::cout << "mVSInputBindDesc.Dimension: " << mVSInputBindDesc.Dimension << std::endl;
            std::cout << "mVSInputBindDesc.Name: " << mVSInputBindDesc.Name << std::endl;
            std::cout << "mVSInputBindDesc.NumSamples: " << mVSInputBindDesc.NumSamples << std::endl;
            std::cout << "mVSInputBindDesc.ReturnType: " << mVSInputBindDesc.ReturnType << std::endl;
            std::cout << "mVSInputBindDesc.Type: " << mVSInputBindDesc.Type << std::endl;
            std::cout << "mVSInputBindDesc.uFlags: " << mVSInputBindDesc.uFlags << std::endl;
        }*/




    }

    if (mPixelShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreatePixelShader(
            mPixelShaderBlob->GetBufferPointer(), mPixelShaderBlob->GetBufferSize(), NULL, &mPixelShader));
        D3DReflect(mPixelShaderBlob->GetBufferPointer(), mPixelShaderBlob->GetBufferSize(), IID_ID3D11ShaderReflection,
                   (void **)&mPixelShaderReflector);
    }

    if (mGeometryShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreateGeometryShader(
            mGeometryShaderBlob->GetBufferPointer(), mGeometryShaderBlob->GetBufferSize(), NULL, &mGeometryShader));
        D3DReflect(mPixelShaderBlob->GetBufferPointer(), mPixelShaderBlob->GetBufferSize(), IID_ID3D11ShaderReflection,
                   (void **)&mPixelShaderReflector);
    }

    mStatus.mVertexShaderCompiled = 1;
    mStatus.mFragmentShaderCompiled = 1;
    mStatus.mGeometryShaderCompiled = 1;
}

void DirectXShaderProgram::bind()
{
    DirectXRenderContext::get()->getD3DDeviceContext()->VSSetShader(mVertexShader, NULL, 0);
    DirectXRenderContext::get()->getD3DDeviceContext()->PSSetShader(mPixelShader, NULL, 0);

    // Replace with calls to bind UniformBuffer I think
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

std::vector<ShaderUniform> DirectXShaderProgram::getUniforms() const
{
    return std::vector<ShaderUniform>();
}

std::vector<ShaderUniform> DirectXShaderProgram::getMaterialUniforms() const
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

void DirectXShaderProgram::setTexture2D(const char *name, int texUnit, void* tex)
{

}

void DirectXShaderProgram::setTexture2Ds(const char *name, const std::vector<int>& texUnits, int count, const std::vector<void*>& texs)
{

}

void DirectXShaderProgram::setBool(int uniformId, bool value)
{
}

void DirectXShaderProgram::setInt(int uniformId, int value)
{
}

void DirectXShaderProgram::setFloat(int uniformId, float value)
{
}

void DirectXShaderProgram::setColor(int uniformId, const Color &color)
{
}

void DirectXShaderProgram::setColor32(int uniformId, const Color32 &color)
{
}

void DirectXShaderProgram::setVec2(int uniformId, const glm::vec2 &vec)
{
}

void DirectXShaderProgram::setVec3(int uniformId, const glm::vec3 &vec)
{
}

void DirectXShaderProgram::setVec4(int uniformId, const glm::vec4 &vec)
{
}

void DirectXShaderProgram::setMat2(int uniformId, const glm::mat2 &mat)
{
}

void DirectXShaderProgram::setMat3(int uniformId, const glm::mat3 &mat)
{
}

void DirectXShaderProgram::setMat4(int uniformId, const glm::mat4 &mat)
{
}

void DirectXShaderProgram::setTexture2D(int uniformId, int texUnit, void *tex)
{
}

void DirectXShaderProgram::setTexture2Ds(int uniformId, const std::vector<int> &texUnits, int count,
                                         const std::vector<void *> &texs)
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

Color32 DirectXShaderProgram::getColor32(const char *name) const
{
    return Color32::black;
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

bool DirectXShaderProgram::getBool(int uniformId) const
{
    return false;
}

int DirectXShaderProgram::getInt(int uniformId) const
{
    return -1;
}

float DirectXShaderProgram::getFloat(int uniformId) const
{
    return 0.0f;
}

Color DirectXShaderProgram::getColor(int uniformId) const
{
    return Color::black;
}

Color32 DirectXShaderProgram::getColor32(int uniformId) const
{
    return Color32::black;
}

glm::vec2 DirectXShaderProgram::getVec2(int uniformId) const
{
    return glm::vec2();
}

glm::vec3 DirectXShaderProgram::getVec3(int uniformId) const
{
    return glm::vec3();
}

glm::vec4 DirectXShaderProgram::getVec4(int uniformId) const
{
    return glm::vec4();
}

glm::mat2 DirectXShaderProgram::getMat2(int uniformId) const
{
    return glm::mat2();
}

glm::mat3 DirectXShaderProgram::getMat3(int uniformId) const
{
    return glm::mat3();
}

glm::mat4 DirectXShaderProgram::getMat4(int uniformId) const
{
    return glm::mat4();
}