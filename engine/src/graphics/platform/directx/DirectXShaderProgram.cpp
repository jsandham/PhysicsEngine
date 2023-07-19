#include "../../../../include/graphics/platform/directx/DirectXShaderProgram.h"
#include "../../../../include/graphics/platform/directx/DirectXError.h"

#include "../../../../include/graphics/platform/directx/DirectXRenderContext.h"

#include "../../../../include/core/Shader.h"

#include <algorithm>
#include <d3dcompiler.h>
#include <iostream>

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
}

DirectXShaderProgram::~DirectXShaderProgram()
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

void DirectXShaderProgram::load(const std::string &name, const std::string &vertex, const std::string &fragment,
                                const std::string &geometry)
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

void DirectXShaderProgram::load(const std::string &name, const std::string &vertex, const std::string &fragment)
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
            Log::error((char *)errorBlob->GetBufferPointer());
            errorBlob->Release();
        }
    }

    // Compile pixel shader shader
    result = D3DCompile(mFragment.data(), mFragment.size(), NULL, NULL, NULL, "PSMain", "ps_5_0", flags, 0,
                        &mPixelShaderBlob, &errorBlob);
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

    std::cout << "mName: " << mName << std::endl;

    ID3D11Device *device = DirectXRenderContext::get()->getD3DDevice();
    assert(device != nullptr);

    if (mVertexShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreateVertexShader(mVertexShaderBlob->GetBufferPointer(),
                                               mVertexShaderBlob->GetBufferSize(), NULL, &mVertexShader));
        ID3D11ShaderReflection *vertexShaderReflector;
        D3DReflect(mVertexShaderBlob->GetBufferPointer(), mVertexShaderBlob->GetBufferSize(),
                   IID_ID3D11ShaderReflection, (void **)&vertexShaderReflector);

        D3D11_SHADER_DESC sd;
        CHECK_ERROR(vertexShaderReflector->GetDesc(&sd));

        std::cout << "Vertex constant buffer count: " << sd.ConstantBuffers << std::endl;

        mVSConstantBuffers.resize(sd.ConstantBuffers);

        // Loop through all constant buffers
        for (unsigned int i = 0; i < sd.ConstantBuffers; i++)
        {
            ID3D11ShaderReflectionConstantBuffer *reflectionBuffer = vertexShaderReflector->GetConstantBufferByIndex(i);
            D3D11_SHADER_BUFFER_DESC constantBufferDesc = {};

            CHECK_ERROR(reflectionBuffer->GetDesc(&constantBufferDesc));

            std::cout << "constantBufferDesc.Name: " << constantBufferDesc.Name << std::endl;
            std::cout << "constantBufferDesc.Size: " << constantBufferDesc.Size << std::endl;
            std::cout << "constantBufferDesc.Type: " << constantBufferDesc.Type << std::endl;
            std::cout << "constantBufferDesc.Variables: " << constantBufferDesc.Variables << std::endl;

            D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
            CHECK_ERROR(vertexShaderReflector->GetResourceBindingDescByName(constantBufferDesc.Name, &inputBindDesc));

            std::cout << "inputBindDesc.BindPoint: " << inputBindDesc.BindPoint << std::endl;

            mVSConstantBuffers[i] = UniformBuffer::create(constantBufferDesc.Size, inputBindDesc.BindPoint);

            for (unsigned int j = 0; j < constantBufferDesc.Variables; j++)
            {
                ID3D11ShaderReflectionVariable *variable = reflectionBuffer->GetVariableByIndex(j);

                D3D11_SHADER_VARIABLE_DESC svd;
                variable->GetDesc(&svd);

                ConstantBufferVariable cbv;
                cbv.mStage = PipelineStage::VS;
                cbv.mUniformId = Shader::uniformToId(svd.Name);
                cbv.mConstantBufferIndex = i;
                cbv.mSize = svd.Size;
                cbv.mOffset = svd.StartOffset;

                mConstantBufferVariables.push_back(cbv);

                std::cout << "svd.Name: " << svd.Name << std::endl;
                std::cout << "svd.Size: " << svd.Size << std::endl;
                std::cout << "svd.StartOffset: " << svd.StartOffset << std::endl;
            }
        }

        vertexShaderReflector->Release();
    }

    if (mPixelShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreatePixelShader(mPixelShaderBlob->GetBufferPointer(), mPixelShaderBlob->GetBufferSize(),
                                              NULL, &mPixelShader));
        ID3D11ShaderReflection *pixelShaderReflector;
        D3DReflect(mPixelShaderBlob->GetBufferPointer(), mPixelShaderBlob->GetBufferSize(), IID_ID3D11ShaderReflection,
                   (void **)&pixelShaderReflector);

        D3D11_SHADER_DESC sd;
        CHECK_ERROR(pixelShaderReflector->GetDesc(&sd));

        std::cout << "Pixel constant buffer count: " << sd.ConstantBuffers << std::endl;

        mPSConstantBuffers.resize(sd.ConstantBuffers);

        // Loop through all constant buffers
        for (unsigned int i = 0; i < sd.ConstantBuffers; i++)
        {
            ID3D11ShaderReflectionConstantBuffer *reflectionBuffer = pixelShaderReflector->GetConstantBufferByIndex(i);
            D3D11_SHADER_BUFFER_DESC constantBufferDesc = {};

            CHECK_ERROR(reflectionBuffer->GetDesc(&constantBufferDesc));

            std::cout << "constantBufferDesc.Name: " << constantBufferDesc.Name << std::endl;
            std::cout << "constantBufferDesc.Size: " << constantBufferDesc.Size << std::endl;
            std::cout << "constantBufferDesc.Type: " << constantBufferDesc.Type << std::endl;
            std::cout << "constantBufferDesc.Variables: " << constantBufferDesc.Variables << std::endl;

            D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
            CHECK_ERROR(pixelShaderReflector->GetResourceBindingDescByName(constantBufferDesc.Name, &inputBindDesc));

            std::cout << "inputBindDesc.BindPoint: " << inputBindDesc.BindPoint << std::endl;

            mPSConstantBuffers[i] = UniformBuffer::create(constantBufferDesc.Size, inputBindDesc.BindPoint);

            for (unsigned int j = 0; j < constantBufferDesc.Variables; j++)
            {
                ID3D11ShaderReflectionVariable *variable = reflectionBuffer->GetVariableByIndex(j);

                D3D11_SHADER_VARIABLE_DESC svd;
                variable->GetDesc(&svd);

                ConstantBufferVariable cbv;
                cbv.mStage = PipelineStage::PS;
                cbv.mUniformId = Shader::uniformToId(svd.Name);
                cbv.mConstantBufferIndex = i;
                cbv.mSize = svd.Size;
                cbv.mOffset = svd.StartOffset;

                mConstantBufferVariables.push_back(cbv);

                std::cout << "svd.Name: " << svd.Name << std::endl;
                std::cout << "svd.Size: " << svd.Size << std::endl;
                std::cout << "svd.StartOffset: " << svd.StartOffset << std::endl;
            }
        }

        pixelShaderReflector->Release();
    }

    if (mGeometryShaderBlob != NULL)
    {
        CHECK_ERROR(device->CreateGeometryShader(mGeometryShaderBlob->GetBufferPointer(),
                                                 mGeometryShaderBlob->GetBufferSize(), NULL, &mGeometryShader));
        ID3D11ShaderReflection *geometryShaderReflector;
        D3DReflect(mGeometryShaderBlob->GetBufferPointer(), mGeometryShaderBlob->GetBufferSize(),
                   IID_ID3D11ShaderReflection, (void **)&geometryShaderReflector);

        D3D11_SHADER_DESC sd;
        CHECK_ERROR(geometryShaderReflector->GetDesc(&sd));

        std::cout << "Geometry constant buffer count: " << sd.ConstantBuffers << std::endl;

        mGSConstantBuffers.resize(sd.ConstantBuffers);

        // Loop through all constant buffers
        for (unsigned int i = 0; i < sd.ConstantBuffers; i++)
        {
            ID3D11ShaderReflectionConstantBuffer *reflectionBuffer =
                geometryShaderReflector->GetConstantBufferByIndex(i);
            D3D11_SHADER_BUFFER_DESC constantBufferDesc = {};

            CHECK_ERROR(reflectionBuffer->GetDesc(&constantBufferDesc));

            std::cout << "constantBufferDesc.Name: " << constantBufferDesc.Name << std::endl;
            std::cout << "constantBufferDesc.Size: " << constantBufferDesc.Size << std::endl;
            std::cout << "constantBufferDesc.Type: " << constantBufferDesc.Type << std::endl;
            std::cout << "constantBufferDesc.Variables: " << constantBufferDesc.Variables << std::endl;

            D3D11_SHADER_INPUT_BIND_DESC inputBindDesc;
            CHECK_ERROR(geometryShaderReflector->GetResourceBindingDescByName(constantBufferDesc.Name, &inputBindDesc));

            std::cout << "inputBindDesc.BindPoint: " << inputBindDesc.BindPoint << std::endl;

            mGSConstantBuffers[i] = UniformBuffer::create(constantBufferDesc.Size, inputBindDesc.BindPoint);

            for (unsigned int j = 0; j < constantBufferDesc.Variables; j++)
            {
                ID3D11ShaderReflectionVariable *variable = reflectionBuffer->GetVariableByIndex(j);

                D3D11_SHADER_VARIABLE_DESC svd;
                variable->GetDesc(&svd);

                ConstantBufferVariable cbv;
                cbv.mStage = PipelineStage::GS;
                cbv.mUniformId = Shader::uniformToId(svd.Name);
                cbv.mConstantBufferIndex = i;
                cbv.mSize = svd.Size;
                cbv.mOffset = svd.StartOffset;

                mConstantBufferVariables.push_back(cbv);

                std::cout << "svd.Name: " << svd.Name << std::endl;
                std::cout << "svd.Size: " << svd.Size << std::endl;
                std::cout << "svd.StartOffset: " << svd.StartOffset << std::endl;
            }
        }

        geometryShaderReflector->Release();
    }

    std::cout << "mConstantBufferVariables" << std::endl;
    for (size_t i = 0; i < mConstantBufferVariables.size(); i++)
    {
        std::cout << "mConstantBufferIndex: " << mConstantBufferVariables[i].mConstantBufferIndex << std::endl;
        std::cout << "mOffset: " << mConstantBufferVariables[i].mOffset << std::endl;
        std::cout << "mSize: " << mConstantBufferVariables[i].mSize << std::endl;
        std::cout << "mStage: " << (int)mConstantBufferVariables[i].mStage << std::endl;
        std::cout << "mUniformId: " << mConstantBufferVariables[i].mUniformId << std::endl;
    }
    std::cout << "" << std::endl;

    mStatus.mVertexShaderCompiled = 1;
    mStatus.mFragmentShaderCompiled = 1;
    mStatus.mGeometryShaderCompiled = 1;
}

void DirectXShaderProgram::bind()
{
    DirectXRenderContext::get()->getD3DDeviceContext()->VSSetShader(mVertexShader, NULL, 0);
    DirectXRenderContext::get()->getD3DDeviceContext()->PSSetShader(mPixelShader, NULL, 0);

    for (size_t i = 0; i < mVSConstantBuffers.size(); i++)
    {
        mVSConstantBuffers[i]->bind(PipelineStage::VS);
    }

    for (size_t i = 0; i < mPSConstantBuffers.size(); i++)
    {
        mPSConstantBuffers[i]->bind(PipelineStage::PS);
    }
}

void DirectXShaderProgram::unbind()
{
    DirectXRenderContext::get()->getD3DDeviceContext()->VSSetShader(NULL, NULL, 0);
    DirectXRenderContext::get()->getD3DDeviceContext()->PSSetShader(NULL, NULL, 0);

    for (size_t i = 0; i < mVSConstantBuffers.size(); i++)
    {
        mVSConstantBuffers[i]->unbind(PipelineStage::VS);
    }

    for (size_t i = 0; i < mPSConstantBuffers.size(); i++)
    {
        mPSConstantBuffers[i]->unbind(PipelineStage::PS);
    }
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
    this->setBool(Shader::uniformToId(name), value);
}

void DirectXShaderProgram::setInt(const char *name, int value)
{
    this->setInt(Shader::uniformToId(name), value);
}

void DirectXShaderProgram::setFloat(const char *name, float value)
{
    this->setFloat(Shader::uniformToId(name), value);
}

void DirectXShaderProgram::setColor(const char *name, const Color &color)
{
    this->setColor(Shader::uniformToId(name), color);
}

void DirectXShaderProgram::setColor32(const char *name, const Color32 &color)
{
    this->setColor32(Shader::uniformToId(name), color);
}

void DirectXShaderProgram::setVec2(const char *name, const glm::vec2 &vec)
{
    this->setVec2(Shader::uniformToId(name), vec);
}

void DirectXShaderProgram::setVec3(const char *name, const glm::vec3 &vec)
{
    this->setVec3(Shader::uniformToId(name), vec);
}

void DirectXShaderProgram::setVec4(const char *name, const glm::vec4 &vec)
{
    this->setVec4(Shader::uniformToId(name), vec);
}

void DirectXShaderProgram::setMat2(const char *name, const glm::mat2 &mat)
{
    this->setMat2(Shader::uniformToId(name), mat);
}

void DirectXShaderProgram::setMat3(const char *name, const glm::mat3 &mat)
{
    this->setMat3(Shader::uniformToId(name), mat);
}

void DirectXShaderProgram::setMat4(const char *name, const glm::mat4 &mat)
{
    this->setMat4(Shader::uniformToId(name), mat);
}

void DirectXShaderProgram::setTexture2D(const char *name, int texUnit, void *tex)
{
    this->setTexture2D(Shader::uniformToId(name), texUnit, tex);
}

void DirectXShaderProgram::setTexture2Ds(const char *name, const std::vector<int> &texUnits, int count,
                                         const std::vector<void *> &texs)
{
    this->setTexture2Ds(Shader::uniformToId(name), texUnits, count, texs);
}

void DirectXShaderProgram::setBool(int uniformId, bool value)
{
    this->setData(uniformId, &value);
}

void DirectXShaderProgram::setInt(int uniformId, int value)
{
    this->setData(uniformId, &value);
}

void DirectXShaderProgram::setFloat(int uniformId, float value)
{
    this->setData(uniformId, &value);
}

void DirectXShaderProgram::setColor(int uniformId, const Color &color)
{
    this->setData(uniformId, &color);
}

void DirectXShaderProgram::setColor32(int uniformId, const Color32 &color)
{
    this->setData(uniformId, &color);
}

void DirectXShaderProgram::setVec2(int uniformId, const glm::vec2 &vec)
{
    this->setData(uniformId, &vec);
}

void DirectXShaderProgram::setVec3(int uniformId, const glm::vec3 &vec)
{
    this->setData(uniformId, &vec);
}

void DirectXShaderProgram::setVec4(int uniformId, const glm::vec4 &vec)
{
    this->setData(uniformId, &vec);
}

void DirectXShaderProgram::setMat2(int uniformId, const glm::mat2 &mat)
{
    this->setData(uniformId, &mat);
}

void DirectXShaderProgram::setMat3(int uniformId, const glm::mat3 &mat)
{
    this->setData(uniformId, &mat);
}

void DirectXShaderProgram::setMat4(int uniformId, const glm::mat4 &mat)
{
    this->setData(uniformId, &mat);
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
    return this->getBool(Shader::uniformToId(name));
}

int DirectXShaderProgram::getInt(const char *name) const
{
    return this->getInt(Shader::uniformToId(name));
}

float DirectXShaderProgram::getFloat(const char *name) const
{
    return this->getFloat(Shader::uniformToId(name));
}

Color DirectXShaderProgram::getColor(const char *name) const
{
    return this->getColor(Shader::uniformToId(name));
}

Color32 DirectXShaderProgram::getColor32(const char *name) const
{
    return this->getColor32(Shader::uniformToId(name));
}

glm::vec2 DirectXShaderProgram::getVec2(const char *name) const
{
    return this->getVec2(Shader::uniformToId(name));
}

glm::vec3 DirectXShaderProgram::getVec3(const char *name) const
{
    return this->getVec3(Shader::uniformToId(name));
}

glm::vec4 DirectXShaderProgram::getVec4(const char *name) const
{
    return this->getVec4(Shader::uniformToId(name));
}

glm::mat2 DirectXShaderProgram::getMat2(const char *name) const
{
    return this->getMat2(Shader::uniformToId(name));
}

glm::mat3 DirectXShaderProgram::getMat3(const char *name) const
{
    return this->getMat3(Shader::uniformToId(name));
}

glm::mat4 DirectXShaderProgram::getMat4(const char *name) const
{
    return this->getMat4(Shader::uniformToId(name));
}

bool DirectXShaderProgram::getBool(int uniformId) const
{
    // bool value = false;
    // this->getData(uniformId, &value);
    // return value;
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

void DirectXShaderProgram::setData(int uniformId, const void *data)
{
    // Step 1: Find which constant buffer this uniform name belongs to..
    // Step 2: Find the offset in the constant buffer where this uniform resides
    // Step 3: Update the value
    ConstantBufferVariable cbv = {};
    for (size_t i = 0; i < mConstantBufferVariables.size(); i++)
    {
        if (mConstantBufferVariables[i].mUniformId == uniformId)
        {
            cbv = mConstantBufferVariables[i];
            break;
        }
    }

    if (cbv.mConstantBufferIndex >= 0)
    {
        switch (cbv.mStage)
        {
        case PipelineStage::VS:
            mVSConstantBuffers[cbv.mConstantBufferIndex]->setData(data, cbv.mOffset, cbv.mSize);
            break;
        case PipelineStage::PS:
            mPSConstantBuffers[cbv.mConstantBufferIndex]->setData(data, cbv.mOffset, cbv.mSize);
            break;
        case PipelineStage::GS:
            mGSConstantBuffers[cbv.mConstantBufferIndex]->setData(data, cbv.mOffset, cbv.mSize);
            break;
        }
    }
}

void DirectXShaderProgram::getData(int uniformId, void *data)
{
    // Step 1: Find which constant buffer this uniform name belongs to..
    // Step 2: Find the offset in the constant buffer where this uniform resides
    // Step 3: Update the value
    ConstantBufferVariable cbv = {};
    for (size_t i = 0; i < mConstantBufferVariables.size(); i++)
    {
        if (mConstantBufferVariables[i].mUniformId == uniformId)
        {
            cbv = mConstantBufferVariables[i];
            break;
        }
    }

    if (cbv.mConstantBufferIndex >= 0)
    {
        switch (cbv.mStage)
        {
        case PipelineStage::VS:
            mVSConstantBuffers[cbv.mConstantBufferIndex]->getData(data, cbv.mOffset, cbv.mSize);
            break;
        case PipelineStage::PS:
            mPSConstantBuffers[cbv.mConstantBufferIndex]->getData(data, cbv.mOffset, cbv.mSize);
            break;
        case PipelineStage::GS:
            mGSConstantBuffers[cbv.mConstantBufferIndex]->getData(data, cbv.mOffset, cbv.mSize);
            break;
        }
    }
}