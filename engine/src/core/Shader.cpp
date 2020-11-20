#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stack>
#include <stdlib.h>
#include <vector>

#include "../../include/core/Log.h"
#include "../../include/core/Shader.h"
#include "../../include/core/shader_load.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Shader::Shader()
{
    mVertexShader = "";
    mFragmentShader = "";
    mGeometryShader = "";

    mAssetId = Guid::INVALID;

    mAllProgramsCompiled = false;
    mActiveProgram = -1;
}

Shader::Shader(const std::vector<char> &data)
{
    deserialize(data);

    mAllProgramsCompiled = false;
    mActiveProgram = -1;
}

Shader::~Shader()
{
}

std::vector<char> Shader::serialize() const
{
    return serialize(mAssetId);
}

std::vector<char> Shader::serialize(Guid assetId) const
{
    ShaderHeader header;
    header.mShaderId = assetId;
    header.mVertexShaderSize = mVertexShader.length();
    header.mGeometryShaderSize = mGeometryShader.length();
    header.mFragmentShaderSize = mFragmentShader.length();
    header.mNumberOfShaderUniforms = mUniforms.size();

    size_t numberOfBytes = sizeof(ShaderHeader) + sizeof(char) * mVertexShader.length() +
                           sizeof(char) * mFragmentShader.length() + sizeof(char) * mGeometryShader.length() +
                           sizeof(ShaderUniform) * mUniforms.size();

    std::vector<char> data(numberOfBytes);

    size_t start1 = 0;
    size_t start2 = start1 + sizeof(ShaderHeader);
    size_t start3 = start2 + sizeof(char) * mVertexShader.length();
    size_t start4 = start3 + sizeof(char) * mGeometryShader.length();

    memcpy(&data[start1], &header, sizeof(ShaderHeader));
    memcpy(&data[start2], mVertexShader.c_str(), sizeof(char) * mVertexShader.length());
    memcpy(&data[start3], mGeometryShader.c_str(), sizeof(char) * mGeometryShader.length());
    memcpy(&data[start4], mFragmentShader.c_str(), sizeof(char) * mFragmentShader.length());

    return data;
}

void Shader::deserialize(const std::vector<char> &data)
{
    size_t start1 = 0;
    size_t start2 = start1 + sizeof(ShaderHeader);

    const ShaderHeader *header = reinterpret_cast<const ShaderHeader *>(&data[start1]);

    mAssetId = header->mShaderId;

    size_t vertexShaderSize = header->mVertexShaderSize;
    size_t geometryShaderSize = header->mGeometryShaderSize;
    size_t fragmentShaderSize = header->mFragmentShaderSize;

    std::vector<char>::const_iterator start = data.begin();
    std::vector<char>::const_iterator end = data.begin();
    start += start2;
    end += start2 + vertexShaderSize;

    mVertexShader = std::string(start, end);

    start += vertexShaderSize;
    end += geometryShaderSize;

    mGeometryShader = std::string(start, end);

    start += geometryShaderSize;
    end += fragmentShaderSize;

    mFragmentShader = std::string(start, end);
}

void Shader::load(const std::string &filepath)
{
    shader_data data;

    if (shader_load(filepath, data))
    {
        this->setVertexShader(data.mVertexShader);
        this->setGeometryShader(data.mGeometryShader);
        this->setFragmentShader(data.mFragmentShader);
    }
    else
    {
        std::string message = "Error: Could not load shader " + filepath + "\n";
        Log::error(message.c_str());
    }
}

void Shader::load(const std::string vertexShader, const std::string fragmentShader, const std::string geometryShader)
{
    this->setVertexShader(vertexShader);
    this->setGeometryShader(geometryShader);
    this->setFragmentShader(fragmentShader);
}

bool Shader::isCompiled() const
{
    return mAllProgramsCompiled;
}

bool Shader::contains(int variant) const
{
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        if (mPrograms[i].mVariant == variant)
        {
            return true;
        }
    }

    return false;
}

void Shader::add(int variant)
{
    if (!contains(variant))
    {
        ShaderProgram program;
        program.mVersion = ShaderVersion::GL430;
        program.mCompiled = false;
        program.mVariant = variant;
        program.mHandle = 0;

        mPrograms.push_back(program);

        mAllProgramsCompiled = false;
    }
}

void Shader::remove(int variant)
{
    int index = -1;
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        if (mPrograms[i].mVariant == variant)
        {
            index = (int)i;
            break;
        }
    }

    if (index != -1)
    {
        mPrograms.erase(mPrograms.begin() + index);
    }
}

void Shader::compile()
{
    // Delete existing shader programs
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        this->unuse();
        Graphics::destroy(mPrograms[i].mHandle);
    }

    // ensure that all shader programs have the default 'None' program variant
    if (!contains(ShaderVariant::None))
    {
        this->add(static_cast<int>(ShaderVariant::None));
    }

    // determine which variants are possible based on keywords found in shader
    const std::vector<std::string> keywords{"DIRECTIONALLIGHT", "SPOTLIGHT", "POINTLIGHT", "HARDSHADOWS",
                                            "SOFTSHADOWS",      "SSAO",      "CASCADE"};

    const std::map<const std::string, ShaderVariant> keywordToVariantMap{
        {"DIRECTIONALLIGHT", ShaderVariant::Directional},
        {"SPOTLIGHT", ShaderVariant::Spot},
        {"POINTLIGHT", ShaderVariant::Point},
        {"HARDSHADOWS", ShaderVariant::HardShadows},
        {"SOFTSHADOWS", ShaderVariant::SoftShadows},
        {"SSAO", ShaderVariant::SSAO},
        {"CASCADE", ShaderVariant::Cascade}};

    std::vector<ShaderVariant> temp;
    for (size_t i = 0; i < keywords.size(); i++)
    {
        if (mVertexShader.find(keywords[i]) != std::string::npos ||
            mGeometryShader.find(keywords[i]) != std::string::npos ||
            mFragmentShader.find(keywords[i]) != std::string::npos)
        {

            std::map<std::string, ShaderVariant>::const_iterator it = keywordToVariantMap.find(keywords[i]);
            if (it != keywordToVariantMap.end())
            {
                temp.push_back(it->second);
            }
        }
    }

    std::set<int> variantsToAdd;
    std::stack<int> stack;
    for (size_t i = 0; i < temp.size(); i++)
    {
        stack.push(temp[i]);
    }

    while (!stack.empty())
    {
        int current = stack.top();
        stack.pop();

        std::set<int>::iterator it = variantsToAdd.find(current);
        if (it == variantsToAdd.end())
        {
            variantsToAdd.insert(current);
        }

        for (size_t i = 0; i < temp.size(); i++)
        {
            if (!(temp[i] & current))
            {
                stack.push(current | temp[i]);
            }
        }
    }

    // add variants from keywords found in shader strings in addition to any variants manually added using 'add' method
    for (std::set<int>::iterator it = variantsToAdd.begin(); it != variantsToAdd.end(); it++)
    {
        this->add(*it);
    }

    // Compile all shader variants
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        std::string version;
        if (mPrograms[i].mVersion == ShaderVersion::GL330)
        {
            version = "#version 330 core\n";
        }
        else
        {
            version = "#version 430 core\n";
        }

        std::string defines;
        if (mPrograms[i].mVariant & ShaderVariant::Directional)
        {
            defines += "#define DIRECTIONALLIGHT\n";
        }
        if (mPrograms[i].mVariant & ShaderVariant::Spot)
        {
            defines += "#define SPOTLIGHT\n";
        }
        if (mPrograms[i].mVariant & ShaderVariant::Point)
        {
            defines += "#define POINTLIGHT\n";
        }
        if (mPrograms[i].mVariant & ShaderVariant::HardShadows)
        {
            defines += "#define HARDSHADOWS\n";
        }
        if (mPrograms[i].mVariant & ShaderVariant::SoftShadows)
        {
            defines += "#define SOFTSHADOWS\n";
        }
        if (mPrograms[i].mVariant & ShaderVariant::SSAO)
        {
            defines += "#define SSAO\n";
        }
        if (mPrograms[i].mVariant & ShaderVariant::Cascade)
        {
            defines += "#define CASCADE\n";
        }

        const std::string vert = version + defines + mVertexShader;
        const std::string geom = version + defines + mGeometryShader;
        const std::string frag = version + defines + mFragmentShader;

        if (mGeometryShader.empty())
        {
            Graphics::compile(vert, frag, "", &(mPrograms[i].mHandle));
        }
        else
        {
            Graphics::compile(vert, frag, geom, &(mPrograms[i].mHandle));
        }

        // Mark shader program compilation successful
        mPrograms[i].mCompiled = true;
    }

    // Mark all shader programs compiled successful
    mAllProgramsCompiled = true;

    // find all uniforms and attributes in shader across all variants
    std::set<std::string> uniformNames;
    for (size_t i = 0; i < mUniforms.size(); i++)
    {
        uniformNames.insert(std::string(mUniforms[i].mName));
    }
    std::set<std::string> attributeNames;
    for (size_t i = 0; i < mAttributes.size(); i++)
    {
        attributeNames.insert(std::string(mAttributes[i].mName));
    }

    // run through all variants and find all uniforms/attributes (and add to sets of known uniforms/attributes if new)
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        GLuint program = mPrograms[i].mHandle;

        std::vector<Uniform> uniforms = Graphics::getUniforms(program);

        for (size_t j = 0; j < uniforms.size(); j++)
        {
            ShaderUniform uniform;
            uniform.mNameLength = (size_t)uniforms[j].nameLength;
            uniform.mSize = (size_t)uniforms[j].size;

            memset(uniform.mData, '\0', 64);
            memset(uniform.mName, '\0', 32);
            memset(uniform.mShortName, '\0', 32);
            memset(uniform.mBlockName, '\0', 32);

            int indexOfBlockChar = -1;
            for (int k = 0; k < uniforms[j].nameLength; k++)
            {
                uniform.mName[k] = uniforms[j].name[k];
                if (uniforms[j].name[k] == '.')
                {
                    indexOfBlockChar = k;
                }
            }

            uniform.mShortName[0] = '\0';
            for (int k = indexOfBlockChar + 1; k < uniforms[j].nameLength; k++)
            {
                uniform.mShortName[k - indexOfBlockChar - 1] = uniforms[j].name[k];
            }

            uniform.mBlockName[0] = '\0';
            for (int k = 0; k < indexOfBlockChar; k++)
            {
                uniform.mBlockName[k] = uniforms[j].name[k];
            }

            uniform.mType = uniforms[j].type;
            uniform.mVariant = mPrograms[i].mVariant;
            uniform.mLocation = findUniformLocation(std::string(uniform.mName), program);

            // only add uniform if it wasnt already in array
            std::set<std::string>::iterator it = uniformNames.find(std::string(uniform.mName));
            if (it == uniformNames.end())
            {
                uniform.mIndex = mUniforms.size();

                mUniforms.push_back(uniform);
                uniformNames.insert(std::string(uniform.mName));
            }
        }

        std::vector<Attribute> attributes = Graphics::getAttributes(program);
        for (size_t j = 0; j < attributes.size(); j++)
        {
            ShaderAttribute attribute;
            for (int k = 0; k < 32; k++)
            {
                attribute.mName[k] = attributes[j].name[k];
            }

            std::set<std::string>::iterator it = attributeNames.find(std::string(attribute.mName));
            if (it == attributeNames.end())
            {
                mAttributes.push_back(attribute);
                attributeNames.insert(std::string(attribute.mName));
            }
        }
    }

    // Finally set camera and light uniform block binding points
    this->setUniformBlock("CamerBlock", 0);
    this->setUniformBlock("LightBlock", 1);
}

void Shader::use(int program)
{
    if (program == -1)
    {
        return;
    }

    mActiveProgram = program;
    Graphics::use(program);
}

void Shader::unuse()
{
    mActiveProgram = -1;
    Graphics::unuse();
}

void Shader::setVertexShader(const std::string vertexShader)
{
    mVertexShader = vertexShader;
    mAllProgramsCompiled = false;
}

void Shader::setGeometryShader(const std::string geometryShader)
{
    mGeometryShader = geometryShader;
    mAllProgramsCompiled = false;
}

void Shader::setFragmentShader(const std::string fragmentShader)
{
    mFragmentShader = fragmentShader;
    mAllProgramsCompiled = false;
}

void Shader::setUniformBlock(const std::string &blockName, int bindingPoint) const
{
    // set uniform block on all shader program
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        Graphics::setUniformBlock(blockName.c_str(), bindingPoint, mPrograms[i].mHandle);
    }
}

int Shader::findUniformLocation(const std::string &name, int program) const
{
    return Graphics::findUniformLocation(name.c_str(), program);
}

int Shader::getProgramFromVariant(int variant) const
{
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        if (mPrograms[i].mVariant == variant)
        {
            return mPrograms[i].mHandle;
        }
    }

    return -1;
}

int Shader::getActiveProgram() const
{
    return mActiveProgram;
}

std::vector<ShaderProgram> Shader::getPrograms() const
{
    return mPrograms;
}

std::vector<ShaderUniform> Shader::getUniforms() const
{
    return mUniforms;
}

std::vector<ShaderAttribute> Shader::getAttributeNames() const
{
    return mAttributes;
}

std::string Shader::getVertexShader() const
{
    return mVertexShader;
}

std::string Shader::getGeometryShader() const
{
    return mGeometryShader;
}

std::string Shader::getFragmentShader() const
{
    return mFragmentShader;
}

void Shader::setBool(const char *name, bool value) const
{
    this->setBool(Graphics::findUniformLocation(name, mActiveProgram), value);
}

void Shader::setInt(const char *name, int value) const
{
    this->setInt(Graphics::findUniformLocation(name, mActiveProgram), value);
}

void Shader::setFloat(const char *name, float value) const
{
    this->setFloat(Graphics::findUniformLocation(name, mActiveProgram), value);
}

void Shader::setColor(const char *name, const Color &color) const
{
    this->setColor(Graphics::findUniformLocation(name, mActiveProgram), color);
}

void Shader::setVec2(const char *name, const glm::vec2 &vec) const
{
    this->setVec2(Graphics::findUniformLocation(name, mActiveProgram), vec);
}

void Shader::setVec3(const char *name, const glm::vec3 &vec) const
{
    this->setVec3(Graphics::findUniformLocation(name, mActiveProgram), vec);
}

void Shader::setVec4(const char *name, const glm::vec4 &vec) const
{
    this->setVec4(Graphics::findUniformLocation(name, mActiveProgram), vec);
}

void Shader::setMat2(const char *name, const glm::mat2 &mat) const
{
    this->setMat2(Graphics::findUniformLocation(name, mActiveProgram), mat);
}

void Shader::setMat3(const char *name, const glm::mat3 &mat) const
{
    this->setMat3(Graphics::findUniformLocation(name, mActiveProgram), mat);
}

void Shader::setMat4(const char *name, const glm::mat4 &mat) const
{
    this->setMat4(Graphics::findUniformLocation(name, mActiveProgram), mat);
}

void Shader::setTexture2D(const char *name, int texUnit, int tex) const
{
    this->setTexture2D(Graphics::findUniformLocation(name, mActiveProgram), texUnit, tex);
}

void Shader::setBool(int nameLocation, bool value) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setBool(nameLocation, (int)value);
    }
}

void Shader::setInt(int nameLocation, int value) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setInt(nameLocation, value);
    }
}

void Shader::setFloat(int nameLocation, float value) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setFloat(nameLocation, value);
    }
}

void Shader::setColor(int nameLocation, const Color &color) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setColor(nameLocation, color);
    }
}

void Shader::setVec2(int nameLocation, const glm::vec2 &vec) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setVec2(nameLocation, vec);
    }
}

void Shader::setVec3(int nameLocation, const glm::vec3 &vec) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setVec3(nameLocation, vec);
    }
}

void Shader::setVec4(int nameLocation, const glm::vec4 &vec) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setVec4(nameLocation, vec);
    }
}

void Shader::setMat2(int nameLocation, const glm::mat2 &mat) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setMat2(nameLocation, mat);
    }
}

void Shader::setMat3(int nameLocation, const glm::mat3 &mat) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setMat3(nameLocation, mat);
    }
}

void Shader::setMat4(int nameLocation, const glm::mat4 &mat) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setMat4(nameLocation, mat);
    }
}

void Shader::setTexture2D(int nameLocation, int texUnit, int tex) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setTexture2D(nameLocation, texUnit, tex);
    }
}

bool Shader::getBool(const char *name) const
{
    return this->getBool(Graphics::findUniformLocation(name, mActiveProgram));
}

int Shader::getInt(const char *name) const
{
    return this->getInt(Graphics::findUniformLocation(name, mActiveProgram));
}

float Shader::getFloat(const char *name) const
{
    return this->getFloat(Graphics::findUniformLocation(name, mActiveProgram));
}

Color Shader::getColor(const char *name) const
{
    return this->getColor(Graphics::findUniformLocation(name, mActiveProgram));
}

glm::vec2 Shader::getVec2(const char *name) const
{
    return this->getVec2(Graphics::findUniformLocation(name, mActiveProgram));
}

glm::vec3 Shader::getVec3(const char *name) const
{
    return this->getVec3(Graphics::findUniformLocation(name, mActiveProgram));
}

glm::vec4 Shader::getVec4(const char *name) const
{
    return this->getVec4(Graphics::findUniformLocation(name, mActiveProgram));
}

glm::mat2 Shader::getMat2(const char *name) const
{
    return this->getMat2(Graphics::findUniformLocation(name, mActiveProgram));
}

glm::mat3 Shader::getMat3(const char *name) const
{
    return this->getMat3(Graphics::findUniformLocation(name, mActiveProgram));
}

glm::mat4 Shader::getMat4(const char *name) const
{
    return this->getMat4(Graphics::findUniformLocation(name, mActiveProgram));
}

int Shader::getTexture2D(const char *name, int texUnit) const
{
    return this->getTexture2D(Graphics::findUniformLocation(name, mActiveProgram), texUnit);
}

bool Shader::getBool(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getBool(nameLocation, mActiveProgram);
    }

    return false;
}

int Shader::getInt(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getInt(nameLocation, mActiveProgram);
    }

    return 0;
}

float Shader::getFloat(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getFloat(nameLocation, mActiveProgram);
    }

    return 0.0f;
}

Color Shader::getColor(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getColor(nameLocation, mActiveProgram);
    }

    return Color(0.0f, 0.0f, 0.0f, 1.0f);
}

glm::vec2 Shader::getVec2(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getVec2(nameLocation, mActiveProgram);
    }

    return glm::vec2(0.0f);
}

glm::vec3 Shader::getVec3(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getVec3(nameLocation, mActiveProgram);
    }

    return glm::vec3(0.0f);
}

glm::vec4 Shader::getVec4(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getVec4(nameLocation, mActiveProgram);
    }

    return glm::vec4(0.0f);
}

glm::mat2 Shader::getMat2(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getMat2(nameLocation, mActiveProgram);
    }

    return glm::mat2(0.0f);
}

glm::mat3 Shader::getMat3(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getMat3(nameLocation, mActiveProgram);
    }

    return glm::mat3(0.0f);
}

glm::mat4 Shader::getMat4(int nameLocation) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getMat4(nameLocation, mActiveProgram);
    }

    return glm::mat4(0.0f);
}

int Shader::getTexture2D(int nameLocation, int texUnit) const
{
    if (mActiveProgram != -1)
    {
        return Graphics::getTexture2D(nameLocation, texUnit, mActiveProgram);
    }

    return -1;
}