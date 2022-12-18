#include <filesystem>

#include "../../include/core/Log.h"
#include "../../include/core/Shader.h"
#include "../../include/core/shader_load.h"
#include "../../include/graphics/Renderer.h"

using namespace PhysicsEngine;

Shader::Shader(World *world, const Id &id) : Asset(world, id)
{
    mSource = "";
    mSourceFilepath = "";
    mVertexShader = "";
    mFragmentShader = "";
    mGeometryShader = "";

    mAllProgramsCompiled = false;
    mActiveProgram = nullptr;

    mShaderSourceLanguage = ShaderSourceLanguage::GLSL;
}

Shader::Shader(World *world, const Guid &guid, const Id &id) : Asset(world, guid, id)
{   
    mSource = "";
    mSourceFilepath = "";
    mVertexShader = "";
    mFragmentShader = "";
    mGeometryShader = "";

    mAllProgramsCompiled = false;
    mActiveProgram = nullptr;

    mShaderSourceLanguage = ShaderSourceLanguage::GLSL;
}

Shader::~Shader()
{
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        delete mPrograms[i];
    }
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
}

bool Shader::isCompiled() const
{
    return mAllProgramsCompiled;
}

void Shader::addVariant(int variantId, const std::set<ShaderMacro> &macros)
{
    mVariantMacroMap[variantId] = macros;
    mAllProgramsCompiled = false;
}

void Shader::removeVariant(int variant)
{
    mVariantMacroMap.erase(variant);
    mAllProgramsCompiled = false;
}

void Shader::preprocess()
{
    for (size_t i = mVariantMacroMap.size(); i < mPrograms.size(); i++)
    {
        delete mPrograms[i];
    }

    size_t count = mPrograms.size();

    mPrograms.resize(mVariantMacroMap.size());
    mVariants.resize(mVariantMacroMap.size());
    for (size_t i = count; i < mPrograms.size(); i++)
    {
        mPrograms[i] = ShaderProgram::create();
    }

    int i = 0;
    for (auto it = mVariantMacroMap.begin(); it != mVariantMacroMap.end(); it++)
    {
        int64_t variant = 0;
        for (auto it2 = it->second.begin(); it2 != it->second.end(); it2++)
        {
            variant |= static_cast<int64_t>(*it2);
        }

        std::string version;
        std::string defines;

        if (variant & static_cast<int64_t>(ShaderMacro::Directional))
        {
            defines += "#define DIRECTIONALLIGHT\n";
        }
        if (variant & static_cast<int64_t>(ShaderMacro::Spot))
        {
            defines += "#define SPOTLIGHT\n";
        }
        if (variant & static_cast<int64_t>(ShaderMacro::Point))
        {
            defines += "#define POINTLIGHT\n";
        }
        if (variant & static_cast<int64_t>(ShaderMacro::HardShadows))
        {
            defines += "#define HARDSHADOWS\n";
        }
        if (variant & static_cast<int64_t>(ShaderMacro::SoftShadows))
        {
            defines += "#define SOFTSHADOWS\n";
        }
        if (variant & static_cast<int64_t>(ShaderMacro::SSAO))
        {
            defines += "#define SSAO\n";
        }
        if (variant & static_cast<int64_t>(ShaderMacro::ShowCascades))
        {
            defines += "#define SHOWCASCADES\n";
        }
        if (variant & static_cast<int64_t>(ShaderMacro::Instancing))
        {
            defines += "#define INSTANCING\n";
        }

        std::string vertexShader = mVertexShader;
        std::string fragmentShader = mFragmentShader;
        std::string geometryShader = mGeometryShader;

        std::string shader;

        size_t pos = vertexShader.find('\n');
        if (pos != std::string::npos)
        {
            version = vertexShader.substr(0, pos + 1);
            shader = vertexShader.substr(pos + 1);
        }

        vertexShader = version + defines + shader;

        pos = fragmentShader.find('\n');
        if (pos != std::string::npos)
        {
            version = fragmentShader.substr(0, pos + 1);
            shader = fragmentShader.substr(pos + 1);
        }

        fragmentShader = version + defines + shader;

        // pos = geometryShader.find('\n');
        // if (pos != std::string::npos)
        //{
        //     version = geometryShader.substr(0, pos + 1);
        //     shader = geometryShader.substr(pos + 1);
        // }

        // geometryShader = version + defines + shader;

        mPrograms[i]->load(vertexShader, fragmentShader);
        mVariants[i] = variant;

        i++;
    }
}

void Shader::compile()
{
    for (size_t i = 0; i < mPrograms.size(); i++)
    {
        mPrograms[i]->compile();
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
        std::vector<ShaderUniform> uniforms = mPrograms[i]->getUniforms();

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
                uniform.mTex = nullptr;
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

void Shader::bind(int variant)
{
    mActiveProgram = getProgramFromVariant(variant);
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->bind();
    }
}

void Shader::unbind()
{
    mActiveProgram->unbind();
    mActiveProgram = nullptr;
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
        Renderer::getRenderer()->setUniformBlock(blockName.c_str(), bindingPoint, *reinterpret_cast<unsigned int*>(mPrograms[i]->getHandle()));
    }
}

int Shader::findUniformLocation(const std::string &name, int program) const
{
    return Renderer::getRenderer()->findUniformLocation(name.c_str(), program);
}

ShaderProgram* Shader::getProgramFromVariant(int64_t variant) const
{
    if (mAllProgramsCompiled)
    {
        for (size_t i = 0; i < mVariants.size(); i++)
        {
            if (mVariants[i] == variant)
            {
                return mPrograms[i];
            }
        }
    }

    return nullptr;
}

ShaderProgram* Shader::getActiveProgram() const
{
    return mActiveProgram;
}

std::vector<ShaderProgram*> Shader::getPrograms() const
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
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setBool(name, value);
    }
}

void Shader::setInt(const char *name, int value) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setInt(name, value);
    }
}

void Shader::setFloat(const char *name, float value) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setFloat(name, value);
    }
}

void Shader::setColor(const char *name, const Color &color) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setColor(name, color);
    }
}

void Shader::setVec2(const char *name, const glm::vec2 &vec) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setVec2(name, vec);
    }
}

void Shader::setVec3(const char *name, const glm::vec3 &vec) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setVec3(name, vec);
    }
}

void Shader::setVec4(const char *name, const glm::vec4 &vec) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setVec4(name, vec);
    }
}

void Shader::setMat2(const char *name, const glm::mat2 &mat) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setMat2(name, mat);
    }
}

void Shader::setMat3(const char *name, const glm::mat3 &mat) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setMat3(name, mat);
    }
}

void Shader::setMat4(const char *name, const glm::mat4 &mat) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setMat4(name, mat);
    }
}

void Shader::setTexture2D(const char *name, int texUnit, TextureHandle* tex) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setTexture2D(name, texUnit, tex);
    }
}

void Shader::setTexture2Ds(const char *name, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setTexture2Ds(name, texUnits, count, texs);
    }
}

void Shader::setBool(int nameLocation, bool value) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setBool(nameLocation, value);
    }
}

void Shader::setInt(int nameLocation, int value) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setInt(nameLocation, value);
    }
}

void Shader::setFloat(int nameLocation, float value) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setFloat(nameLocation, value);
    }
}

void Shader::setColor(int nameLocation, const Color &color) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setColor(nameLocation, color);
    }
}

void Shader::setVec2(int nameLocation, const glm::vec2 &vec) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setVec2(nameLocation, vec);
    }
}

void Shader::setVec3(int nameLocation, const glm::vec3 &vec) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setVec3(nameLocation, vec);
    }
}

void Shader::setVec4(int nameLocation, const glm::vec4 &vec) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setVec4(nameLocation, vec);
    }
}

void Shader::setMat2(int nameLocation, const glm::mat2 &mat) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setMat2(nameLocation, mat);
    }
}

void Shader::setMat3(int nameLocation, const glm::mat3 &mat) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setMat3(nameLocation, mat);
    }
}

void Shader::setMat4(int nameLocation, const glm::mat4 &mat) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setMat4(nameLocation, mat);
    }
}

void Shader::setTexture2D(int nameLocation, int texUnit, TextureHandle* tex) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setTexture2D(nameLocation, texUnit, tex);
    }
}

void Shader::setTexture2Ds(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<TextureHandle*>& texs) const
{
    if (mActiveProgram != nullptr)
    {
        mActiveProgram->setTexture2Ds(nameLocation, texUnits, count, texs);
    }
}

bool Shader::getBool(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getBool(name) : false;
}

int Shader::getInt(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getInt(name) : -1;
}

float Shader::getFloat(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getFloat(name) : 0.0f;
}

Color Shader::getColor(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getColor(name) : Color::black;
}

glm::vec2 Shader::getVec2(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getVec2(name) : glm::vec2();
}

glm::vec3 Shader::getVec3(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getVec3(name) : glm::vec3();
}

glm::vec4 Shader::getVec4(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getVec4(name) : glm::vec4();
}

glm::mat2 Shader::getMat2(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getMat2(name) : glm::mat2();
}

glm::mat3 Shader::getMat3(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getMat3(name) : glm::mat3();
}

glm::mat4 Shader::getMat4(const char *name) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getMat4(name) : glm::mat4();
}

int Shader::getTexture2D(const char *name, int texUnit) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getTexture2D(name, texUnit) : -1;
}

bool Shader::getBool(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getBool(nameLocation) : false;
}

int Shader::getInt(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getInt(nameLocation) : 0;
}

float Shader::getFloat(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getFloat(nameLocation) : 0.0f;
}

Color Shader::getColor(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getColor(nameLocation) : Color::black;
}

glm::vec2 Shader::getVec2(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getVec2(nameLocation) : glm::vec2(0.0f);
}

glm::vec3 Shader::getVec3(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getVec3(nameLocation) : glm::vec3(0.0f);
}

glm::vec4 Shader::getVec4(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getVec4(nameLocation) : glm::vec4(0.0f);
}

glm::mat2 Shader::getMat2(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getMat2(nameLocation) : glm::mat2(0.0f);
}

glm::mat3 Shader::getMat3(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getMat3(nameLocation) : glm::mat3(0.0f);
}

glm::mat4 Shader::getMat4(int nameLocation) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getMat4(nameLocation) : glm::mat4(0.0f);
}

int Shader::getTexture2D(int nameLocation, int texUnit) const
{
    return mActiveProgram != nullptr ? mActiveProgram->getTexture2D(nameLocation, texUnit) : -1;
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