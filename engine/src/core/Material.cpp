#include "../../include/core/Material.h"
#include "../../include/core/World.h"
#include "../../include/core/Log.h"
#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Material::Material(World *world, const Id &id) : Asset(world, id)
{
    mShaderGuid = Guid::INVALID;
    mRenderQueue = RenderQueue::Opaque;

    mShaderChanged = true;
    mTextureChanged = true;
    mEnableInstancing = false;
}

Material::Material(World *world, const Guid &guid, const Id &id) : Asset(world, guid, id)
{
    mShaderGuid = Guid::INVALID;
    mRenderQueue = RenderQueue::Opaque;

    mShaderChanged = true;
    mTextureChanged = true;
    mEnableInstancing = false;
}

Material::~Material()
{
}

void Material::serialize(YAML::Node &out) const
{
    Asset::serialize(out);

    out["shaderId"] = mShaderGuid;
    out["renderQueue"] = mRenderQueue;
    out["enableInstancing"] = mEnableInstancing;

    for (size_t i = 0; i < mUniforms.size(); i++)
    {
        out[mUniforms[i].mName] = mUniforms[i];
    }
}

void Material::deserialize(const YAML::Node &in)
{
    Asset::deserialize(in);

    mShaderGuid = YAML::getValue<Guid>(in, "shaderId");
    mRenderQueue = YAML::getValue<RenderQueue>(in, "renderQueue");
    mEnableInstancing = YAML::getValue<bool>(in, "enableInstancing");

    mUniforms.clear();

    int index = 0;
    for (YAML::const_iterator it = in.begin(); it != in.end(); ++it)
    {
        index++;

        if (index <= 7)
        {
            continue;
        }

        ShaderUniform uniform = YAML::getValue<ShaderUniform>(in, it->first.as<std::string>());
        uniform.mName = it->first.as<std::string>();

        mUniforms.push_back(uniform);
    }

    if (mShaderGuid.isValid())
    {
        mWorld->cacheMaterialUniforms(getGuid(), mShaderGuid, mUniforms);
    }

    mShaderChanged = true;
    mTextureChanged = true;
}

int Material::getType() const
{
    return PhysicsEngine::MATERIAL_TYPE;
}

std::string Material::getObjectName() const
{
    return PhysicsEngine::MATERIAL_NAME;
}

void Material::apply()
{
    Shader *shader = mWorld->getAssetByGuid<Shader>(mShaderGuid);

    assert(shader != nullptr);

    //Renderer::getRenderer()->applyMaterial(mUniforms, shader->getActiveProgram());
    ShaderProgram *shaderProgram = shader->getActiveProgram();
    
    int textureUnit = 0;
    for (size_t i = 0; i < mUniforms.size(); i++)
    {
        int location = shaderProgram->findUniformLocation(mUniforms[i].mName);

        assert(location != -1);

        if (mUniforms[i].mType == ShaderUniformType::Sampler2D)
        {
            if (mUniforms[i].mTex != nullptr)
            {
                shaderProgram->setTexture2D(location, textureUnit, mUniforms[i].mTex);
            }
            else
            {
                shaderProgram->setTexture2D(location, textureUnit, nullptr);
            }

            textureUnit++;
        }
        else if (mUniforms[i].mType == ShaderUniformType::Int)
        {
            shaderProgram->setInt(location, *reinterpret_cast<const int *>(mUniforms[i].mData));
        }
        else if (mUniforms[i].mType == ShaderUniformType::Float)
        {
            shaderProgram->setFloat(location, *reinterpret_cast<const float *>(mUniforms[i].mData));
        }
        else if (mUniforms[i].mType == ShaderUniformType::Vec2)
        {
            shaderProgram->setVec2(location, *reinterpret_cast<const glm::vec2 *>(mUniforms[i].mData));
        }
        else if (mUniforms[i].mType == ShaderUniformType::Vec3)
        {
            shaderProgram->setVec3(location, *reinterpret_cast<const glm::vec3 *>(mUniforms[i].mData));
        }
        else if (mUniforms[i].mType == ShaderUniformType::Vec4)
        {
            shaderProgram->setVec4(location, *reinterpret_cast<const glm::vec4 *>(mUniforms[i].mData));
        }
    }
}

void Material::onShaderChanged()
{
    Shader *shader = mWorld->getAssetByGuid<Shader>(mShaderGuid);

    if (shader == nullptr)
    {
        return;
    }

    if (!shader->isCompiled())
    {
        Log::error("Must compile shader before calling onShaderChanged\n");
        return;
    }

    std::vector<ShaderUniform> newUniforms = shader->getMaterialUniforms();

    // Attempt to copy cached uniform data to new uniforms
    std::vector<ShaderUniform> cachedUniforms = mWorld->getCachedMaterialUniforms(getGuid(), mShaderGuid);
    if (newUniforms.size() == cachedUniforms.size())
    {
        for (size_t i = 0; i < newUniforms.size(); i++)
        {
            if (newUniforms[i].mType == cachedUniforms[i].mType && newUniforms[i].mName == cachedUniforms[i].mName)
            {
                newUniforms[i].mUniformId = cachedUniforms[i].mUniformId;
                newUniforms[i].mTex = cachedUniforms[i].mTex;
                memcpy(newUniforms[i].mData, cachedUniforms[i].mData, 64);
            }
        }
    }

    mUniforms = newUniforms;

    mShaderChanged = false;
}

void Material::onTextureChanged()
{
    // Find all texture handles
    for (size_t i = 0; i < mUniforms.size(); i++)
    {
        if (mUniforms[i].mType == ShaderUniformType::Sampler2D)
        {
            Texture2D *texture = mWorld->getAssetByGuid<Texture2D>(*reinterpret_cast<Guid *>(mUniforms[i].mData));
            if (texture != nullptr)
            {
                mUniforms[i].mTex = texture->getNativeGraphics();
            }
            else
            {
                mUniforms[i].mTex = nullptr;
            }
        }
    }

    mTextureChanged = false;
}

bool Material::hasShaderChanged() const
{
    return mShaderChanged;
}

bool Material::hasTextureChanged() const
{
    return mTextureChanged;
}

void Material::setShaderId(const Guid& shaderId)
{
    // If current shader on material was valid, cache its uniforms before setting new shader
    if (mShaderGuid.isValid())
    {
        mWorld->cacheMaterialUniforms(getGuid(), mShaderGuid, mUniforms);
    }

    mShaderGuid = shaderId;
    mShaderChanged = true;
}

Guid Material::getShaderId() const
{
    return mShaderGuid;
}

std::vector<ShaderUniform> Material::getUniforms() const
{
    return mUniforms;
}

void Material::setBool(const std::string &name, bool value)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy((void *)mUniforms[index].mData, &value, sizeof(bool));
    }
}

void Material::setInt(const std::string &name, int value)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy((void *)mUniforms[index].mData, &value, sizeof(int));
    }
}

void Material::setFloat(const std::string &name, float value)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Float)
    {
        memcpy((void *)mUniforms[index].mData, &value, sizeof(float));
    }
}

void Material::setColor(const std::string &name, const Color &color)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Color)
    {
        memcpy((void *)mUniforms[index].mData, &color, sizeof(Color));
    }
}

void Material::setVec2(const std::string &name, const glm::vec2 &vec)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec2)
    {
        memcpy((void *)mUniforms[index].mData, &vec, sizeof(glm::vec2));
    }
}

void Material::setVec3(const std::string &name, const glm::vec3 &vec)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec3)
    {
        memcpy((void *)mUniforms[index].mData, &vec, sizeof(glm::vec3));
    }
}

void Material::setVec4(const std::string &name, const glm::vec4 &vec)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec4)
    {
        memcpy((void *)mUniforms[index].mData, &vec, sizeof(glm::vec4));
    }
}

void Material::setMat2(const std::string &name, const glm::mat2 &mat)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat2)
    {
        memcpy((void *)mUniforms[index].mData, &mat, sizeof(glm::mat2));
    }
}

void Material::setMat3(const std::string &name, const glm::mat3 &mat)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat3)
    {
        memcpy((void *)mUniforms[index].mData, &mat, sizeof(glm::mat3));
    }
}

void Material::setMat4(const std::string &name, const glm::mat4 &mat)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat4)
    {
        memcpy((void *)mUniforms[index].mData, &mat, sizeof(glm::mat4));
    }
}

void Material::setTexture(const std::string &name, const Guid &textureId)
{
    int index = findIndexOfUniform(name);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Sampler2D)
    {
        memcpy((void *)mUniforms[index].mData, &textureId, sizeof(Guid));
    }

    mTextureChanged = true;
}

void Material::setBool(int uniformId, bool value)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy((void *)mUniforms[index].mData, &value, sizeof(bool));
    }
}

void Material::setInt(int uniformId, int value)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy((void *)mUniforms[index].mData, &value, sizeof(int));
    }
}

void Material::setFloat(int uniformId, float value)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Float)
    {
        memcpy((void *)mUniforms[index].mData, &value, sizeof(float));
    }
}

void Material::setColor(int uniformId, const Color &color)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Color)
    {
        memcpy((void *)mUniforms[index].mData, &color, sizeof(Color));
    }
}

void Material::setVec2(int uniformId, const glm::vec2 &vec)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec2)
    {
        memcpy((void *)mUniforms[index].mData, &vec, sizeof(glm::vec2));
    }
}

void Material::setVec3(int uniformId, const glm::vec3 &vec)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec3)
    {
        memcpy((void *)mUniforms[index].mData, &vec, sizeof(glm::vec3));
    }
}

void Material::setVec4(int uniformId, const glm::vec4 &vec)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec4)
    {
        memcpy((void *)mUniforms[index].mData, &vec, sizeof(glm::vec4));
    }
}

void Material::setMat2(int uniformId, const glm::mat2 &mat)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat2)
    {
        memcpy((void *)mUniforms[index].mData, &mat, sizeof(glm::mat2));
    }
}

void Material::setMat3(int uniformId, const glm::mat3 &mat)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat3)
    {
        memcpy((void *)mUniforms[index].mData, &mat, sizeof(glm::mat3));
    }
}

void Material::setMat4(int uniformId, const glm::mat4 &mat)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat4)
    {
        memcpy((void *)mUniforms[index].mData, &mat, sizeof(glm::mat4));
    }
}

void Material::setTexture(int uniformId, const Guid &textureId)
{
    int index = findIndexOfUniform(uniformId);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Sampler2D)
    {
        memcpy((void *)mUniforms[index].mData, &textureId, sizeof(textureId));
    }

    mTextureChanged = true;
}

bool Material::getBool(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    bool value = false;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy(&value, mUniforms[index].mData, sizeof(bool));
    }

    return value;
}

int Material::getInt(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    int value = 0;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy(&value, mUniforms[index].mData, sizeof(int));
    }

    return value;
}

float Material::getFloat(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    float value = 0.0f;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Float)
    {
        memcpy(&value, mUniforms[index].mData, sizeof(float));
    }

    return value;
}

Color Material::getColor(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    Color color = Color(0, 0, 0, 255);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Color)
    {
        memcpy(&color, mUniforms[index].mData, sizeof(Color));
    }

    return color;
}

glm::vec2 Material::getVec2(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    glm::vec2 vec = glm::vec2(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec2)
    {
        memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec2));
    }

    return vec;
}

glm::vec3 Material::getVec3(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    glm::vec3 vec = glm::vec3(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec3)
    {
        memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec3));
    }

    return vec;
}

glm::vec4 Material::getVec4(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    glm::vec4 vec = glm::vec4(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec4)
    {
        memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec4));
    }

    return vec;
}

glm::mat2 Material::getMat2(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    glm::mat2 mat = glm::mat2(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat2)
    {
        memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat2));
    }

    return mat;
}

glm::mat3 Material::getMat3(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    glm::mat3 mat = glm::mat3(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat3)
    {
        memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat3));
    }

    return mat;
}

glm::mat4 Material::getMat4(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    glm::mat4 mat = glm::mat4(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat4)
    {
        memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat4));
    }

    return mat;
}

Guid Material::getTexture(const std::string &name) const
{
    int index = findIndexOfUniform(name);
    Guid textureId = Guid::INVALID;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Sampler2D)
    {
        memcpy(&textureId, mUniforms[index].mData, sizeof(Guid));
    }

    return textureId;
}

bool Material::getBool(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    bool value = false;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy(&value, mUniforms[index].mData, sizeof(bool));
    }

    return value;
}

int Material::getInt(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    int value = 0;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Int)
    {
        memcpy(&value, mUniforms[index].mData, sizeof(int));
    }

    return value;
}

float Material::getFloat(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    float value = 0.0f;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Float)
    {
        memcpy(&value, mUniforms[index].mData, sizeof(float));
    }

    return value;
}

Color Material::getColor(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    Color color = Color(0, 0, 0, 255);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Color)
    {
        memcpy(&color, mUniforms[index].mData, sizeof(Color));
    }

    return color;
}

glm::vec2 Material::getVec2(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    glm::vec2 vec = glm::vec2(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec2)
    {
        memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec2));
    }

    return vec;
}

glm::vec3 Material::getVec3(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    glm::vec3 vec = glm::vec3(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec3)
    {
        memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec3));
    }

    return vec;
}

glm::vec4 Material::getVec4(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    glm::vec4 vec = glm::vec4(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Vec4)
    {
        memcpy(&vec, mUniforms[index].mData, sizeof(glm::vec4));
    }

    return vec;
}

glm::mat2 Material::getMat2(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    glm::mat2 mat = glm::mat2(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat2)
    {
        memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat2));
    }

    return mat;
}

glm::mat3 Material::getMat3(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    glm::mat3 mat = glm::mat3(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat3)
    {
        memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat3));
    }

    return mat;
}

glm::mat4 Material::getMat4(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    glm::mat4 mat = glm::mat4(0.0f);
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Mat4)
    {
        memcpy(&mat, mUniforms[index].mData, sizeof(glm::mat4));
    }

    return mat;
}

Guid Material::getTexture(int uniformId) const
{
    int index = findIndexOfUniform(uniformId);
    Guid textureId = Guid::INVALID;
    if (index != -1 && mUniforms[index].mType == ShaderUniformType::Sampler2D)
    {
        memcpy(&textureId, mUniforms[index].mData, sizeof(Guid));
    }

    return textureId;
}

std::vector<Guid> Material::getTextures() const
{
    std::vector<Guid> textures;
    for (size_t i = 0; i < mUniforms.size(); i++)
    {
        if (mUniforms[i].mType == ShaderUniformType::Sampler2D)
        {
            Guid textureId = Guid::INVALID;
            memcpy(&textureId, mUniforms[i].mData, sizeof(Guid));

            textures.push_back(textureId);
        }
    }

    return textures;
}

int Material::findIndexOfUniform(const std::string &name) const
{
    for (size_t i = 0; i < mUniforms.size(); i++)
    {
        if (name == mUniforms[i].mName)
        {
            return (int)i;
        }
    }

    return -1;
}

int Material::findIndexOfUniform(unsigned int uniformId) const
{
    for (size_t i = 0; i < mUniforms.size(); i++)
    {
        if (uniformId == mUniforms[i].mUniformId)
        {
            return (int)i;
        }
    }

    return -1;
}