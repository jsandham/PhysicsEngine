#include <filesystem>

#include "../../include/core/Log.h"
#include "../../include/core/Shader.h"
#include "../../include/core/shader_load.h"
#include "../../include/graphics/Graphics.h"

using namespace PhysicsEngine;

Shader::Shader(World *world) : Asset(world)
{
    mSource = "";
    mSourceFilepath = "";
    mVertexShader = "";
    mFragmentShader = "";
    mGeometryShader = "";

    mAllProgramsCompiled = false;
    mActiveProgram = -1;

    mShaderSourceLanguage = ShaderSourceLanguage::GLSL;
}

Shader::Shader(World *world, Id id) : Asset(world, id)
{   
    mSource = "";
    mSourceFilepath = "";
    mVertexShader = "";
    mFragmentShader = "";
    mGeometryShader = "";

    mAllProgramsCompiled = false;
    mActiveProgram = -1;

    mShaderSourceLanguage = ShaderSourceLanguage::GLSL;
}

Shader::~Shader()
{
}

void Shader::serialize(YAML::Node &out) const
{
    Asset::serialize(out);

    out["shaderSourceLanguage"] = mShaderSourceLanguage;
    out["source"] = mSource;
    out["variants"] = mVariantMacroMap;
}

void Shader::deserialize(const YAML::Node &in)
{
    Asset::deserialize(in);

    mShaderSourceLanguage = YAML::getValue<ShaderSourceLanguage>(in, "shaderSourceLanguage");
    mVariantMacroMap = YAML::getValue<std::unordered_map<int, std::set<ShaderMacro>>>(in, "variants");
    mSource = YAML::getValue<std::string>(in, "source");
    mSourceFilepath = YAML::getValue<std::string>(in, "sourceFilepath"); // dont serialize out

    ShaderCreationAttrib attrib;
    attrib.mName = mName;
    attrib.mSourceFilepath = mSourceFilepath;
    attrib.mSourceLanguage = mShaderSourceLanguage;
    attrib.mVariantMacroMap = mVariantMacroMap;

    load(attrib);
}

int Shader::getType() const
{
    return PhysicsEngine::SHADER_TYPE;
}

std::string Shader::getObjectName() const
{
    return PhysicsEngine::SHADER_NAME;
}

void Shader::load(const ShaderCreationAttrib& attrib)
{
    if (attrib.mSourceFilepath.empty()){ return; }

    shader_data data;

    if (shader_load(attrib.mSourceFilepath, data))
    {
        this->setVertexShader(data.mVertexShader);
        this->setGeometryShader(data.mGeometryShader);
        this->setFragmentShader(data.mFragmentShader);
    }
    else
    {
        std::string message = "Error: Could not load shader " + attrib.mName + " \n";
        Log::error(message.c_str());
    }

    mName = attrib.mName;
    mShaderSourceLanguage = attrib.mSourceLanguage;
    mVariantMacroMap = attrib.mVariantMacroMap;

    std::filesystem::path temp = attrib.mSourceFilepath;
    mSource = temp.filename().string();

    mPrograms.resize(attrib.mVariantMacroMap.size());
}

bool Shader::isCompiled() const
{
    return mAllProgramsCompiled;
}

void Shader::addVariant(int variantId, const std::set<ShaderMacro> &macros)
{
    mVariantMacroMap[variantId] = macros;

    mPrograms.resize(mVariantMacroMap.size());
}

void Shader::preprocess()
{
    int i = 0;
    for (auto it = mVariantMacroMap.begin(); it != mVariantMacroMap.end(); it++)
    {
        mPrograms[i].mVertexShader = mVertexShader;
        mPrograms[i].mFragmentShader = mFragmentShader;
        mPrograms[i].mGeometryShader = mGeometryShader;

        int64_t variant = 0;
        for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
        {
            variant |= static_cast<int64_t>(*it2);
        }

        mPrograms[i].mVariant = variant;
        mPrograms[i].mHandle = 0;

        Graphics::preprocess(mPrograms[i].mVertexShader,
                             mPrograms[i].mFragmentShader,
                             mPrograms[i].mGeometryShader,
                             mPrograms[i].mVariant);

        i++;
    }
}

void Shader::compile()
{
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        Graphics::destroy(mPrograms[i].mHandle);

        Graphics::compile(mName, mPrograms[i].mVertexShader, mPrograms[i].mFragmentShader, mPrograms[i].mGeometryShader,
                          &mPrograms[i].mHandle, mPrograms[i].mStatus);
    }

    mAllProgramsCompiled = true;

    // Finally set camera and light uniform block binding points
    this->setUniformBlock("CameraBlock", 0);
    this->setUniformBlock("LightBlock", 1);

    // find all uniforms and attributes in shader across all variants
    std::set<std::string> uniformNames;
    std::set<std::string> attributeNames;

    mUniforms.clear();
    mMaterialUniforms.clear();

    // run through all variants and find all uniforms/attributes (and add to sets of known uniforms/attributes if new)
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        unsigned int program = mPrograms[i].mHandle;

        std::vector<ShaderUniform> uniforms = Graphics::getShaderUniforms(program);

        for (size_t j = 0; j < uniforms.size(); j++)
        {
            std::string name = uniforms[j].mName;
            ShaderUniformType type = uniforms[j].mType;

            std::set<std::string>::iterator it = uniformNames.find(name);
            if (it == uniformNames.end())
            {
                ShaderUniform uniform;
                uniform.mName = name;
                uniform.mType = type;
                uniform.mTex = -1;
                uniform.mUniformId = Shader::uniformToId(name.c_str());
                memset(uniform.mData, '\0', 64);

                mUniforms.push_back(uniform);
                uniformNames.insert(uniform.mName);

                if (name.find("material") != std::string::npos)
                {
                    mMaterialUniforms.push_back(uniform);
                }
            }
        }
    }
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

void Shader::setVertexShader(const std::string &vertexShader)
{
    mVertexShader = vertexShader;
    mAllProgramsCompiled = false;
}

void Shader::setGeometryShader(const std::string &geometryShader)
{
    mGeometryShader = geometryShader;
    mAllProgramsCompiled = false;
}

void Shader::setFragmentShader(const std::string &fragmentShader)
{
    mFragmentShader = fragmentShader;
    mAllProgramsCompiled = false;
}

std::string Shader::getSource() const
{
    return mSource;
}

std::string Shader::getSourceFilepath() const
{
    return mSourceFilepath;
}

ShaderSourceLanguage Shader::getSourceLanguage() const
{
    return mShaderSourceLanguage;
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

int Shader::getProgramFromVariant(int64_t variant) const
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

std::vector<ShaderUniform> Shader::getMaterialUniforms() const
{
    return mMaterialUniforms;
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

void Shader::setTexture2Ds(const char *name, int *texUnits, int count, int *texs) const
{
    this->setTexture2Ds(Graphics::findUniformLocation(name, mActiveProgram), texUnits, count, texs);
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

void Shader::setTexture2Ds(int nameLocation, int *texUnits, int count, int *texs) const
{
    if (mActiveProgram != -1)
    {
        Graphics::setTexture2Ds(nameLocation, texUnits, count, texs);
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

unsigned int Shader::uniformToId(const char *uniform)
{
    unsigned int hash = 5381;
    int c;

    c = *uniform++;
    while (c != 0)
    {
        hash = ((hash << 5) + hash) + c;
        c = *uniform++;
    }
    
    return hash;
}