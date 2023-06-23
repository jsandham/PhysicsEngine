#ifndef SHADERPROGRAM_H__
#define SHADERPROGRAM_H__

#include "TextureHandle.h"
#include "RenderTextureHandle.h"

#include "../core/Color.h"
#include <glm/glm.hpp>

namespace PhysicsEngine
{
    enum class ShaderUniformType
    {
        Int = 0,
        Float = 1,
        Color = 2,
        Vec2 = 3,
        Vec3 = 4,
        Vec4 = 5,
        Mat2 = 6,
        Mat3 = 7,
        Mat4 = 8,
        Sampler2D = 9,
        SamplerCube = 10,
        Invalid = 11
    };

    struct ShaderUniform
    {
        char mData[64];
        std::string mName;       // variable name (including block name if applicable)
        ShaderUniformType mType; // type of the uniform (float, vec3 or mat4, etc)
        void *mTex; // if data stores a texture id, this is the texture handle
        int mUniformId; // integer hash of uniform name

        std::string getShortName() const
        {
            size_t pos = mName.find_first_of('.');
            return mName.substr(pos + 1);
        }
    };

    struct ShaderAttribute
    {
        std::string mName;
    };

    struct ShaderStatus
    {
        char mVertexCompileLog[512];
        char mFragmentCompileLog[512];
        char mGeometryCompileLog[512];
        char mLinkLog[512];
        int mVertexShaderCompiled;
        int mFragmentShaderCompiled;
        int mGeometryShaderCompiled;
        int mShaderLinked;
    };

	class ShaderProgram
	{
      protected:
        std::string mName;
        std::string mVertex;
        std::string mFragment;
        std::string mGeometry;

        std::vector<ShaderUniform> mUniforms;
        std::vector<ShaderUniform> mMaterialUniforms;
        std::vector<ShaderAttribute> mAttributes;

        ShaderStatus mStatus;

      public:
        ShaderProgram();
        ShaderProgram(const ShaderProgram &other) = delete;
        ShaderProgram &operator=(const ShaderProgram &other) = delete;
        virtual ~ShaderProgram() = 0;

        std::string getVertexShader() const;
        std::string getFragmentShader() const;
        std::string getGeometryShader() const;
        ShaderStatus getStatus() const;

        virtual void load(const std::string& name, const std::string &vertex, const std::string &fragment, const std::string &geometry) = 0;
        virtual void load(const std::string& name, const std::string &vertex, const std::string &fragment) = 0;
        virtual void compile() = 0;
        virtual void bind() = 0;
        virtual void unbind() = 0;

        virtual std::vector<ShaderUniform> getUniforms() const = 0;
        virtual std::vector<ShaderUniform> getMaterialUniforms() const = 0;
        virtual std::vector<ShaderAttribute> getAttributes() const = 0;

        virtual void setBool(const char* name, bool value) = 0;
        virtual void setInt(const char *name, int value) = 0;
        virtual void setFloat(const char *name, float value) = 0;
        virtual void setColor(const char *name, const Color &color) = 0;
        virtual void setColor32(const char *name, const Color32 &color) = 0;
        virtual void setVec2(const char *name, const glm::vec2 &vec) = 0;
        virtual void setVec3(const char *name, const glm::vec3 &vec) = 0;
        virtual void setVec4(const char *name, const glm::vec4 &vec) = 0;
        virtual void setMat2(const char *name, const glm::mat2 &mat) = 0;
        virtual void setMat3(const char *name, const glm::mat3 &mat) = 0;
        virtual void setMat4(const char *name, const glm::mat4 &mat) = 0;
        virtual void setTexture2D(const char *name, int texUnit, void *tex) = 0;
        virtual void setTexture2Ds(const char *name, const std::vector<int> &texUnits, int count,
                                   const std::vector<void *> &texs) = 0;

        virtual void setBool(int uniformId, bool value) = 0;
        virtual void setInt(int uniformId, int value) = 0;
        virtual void setFloat(int uniformId, float value) = 0;
        virtual void setColor(int uniformId, const Color &color) = 0;
        virtual void setColor32(int uniformId, const Color32 &color) = 0;
        virtual void setVec2(int uniformId, const glm::vec2 &vec) = 0;
        virtual void setVec3(int uniformId, const glm::vec3 &vec) = 0;
        virtual void setVec4(int uniformId, const glm::vec4 &vec) = 0;
        virtual void setMat2(int uniformId, const glm::mat2 &mat) = 0;
        virtual void setMat3(int uniformId, const glm::mat3 &mat) = 0;
        virtual void setMat4(int uniformId, const glm::mat4 &mat) = 0;
        virtual void setTexture2D(int uniformId, int texUnit, void *tex) = 0;
        virtual void setTexture2Ds(int uniformId, const std::vector<int> &texUnits, int count,
                                   const std::vector<void *> &texs) = 0;
        
        virtual bool getBool(const char *name) const = 0;
        virtual int getInt(const char *name) const = 0;
        virtual float getFloat(const char *name) const = 0;
        virtual Color getColor(const char *name) const = 0;
        virtual Color32 getColor32(const char *name) const = 0;
        virtual glm::vec2 getVec2(const char *name) const = 0;
        virtual glm::vec3 getVec3(const char *name) const = 0;
        virtual glm::vec4 getVec4(const char *name) const = 0;
        virtual glm::mat2 getMat2(const char *name) const = 0;
        virtual glm::mat3 getMat3(const char *name) const = 0;
        virtual glm::mat4 getMat4(const char *name) const = 0;
        
        virtual bool getBool(int uniformId) const = 0;
        virtual int getInt(int uniformId) const = 0;
        virtual float getFloat(int uniformId) const = 0;
        virtual Color getColor(int uniformId) const = 0;
        virtual Color32 getColor32(int uniformId) const = 0;
        virtual glm::vec2 getVec2(int uniformId) const = 0;
        virtual glm::vec3 getVec3(int uniformId) const = 0;
        virtual glm::vec4 getVec4(int uniformId) const = 0;
        virtual glm::mat2 getMat2(int uniformId) const = 0;
        virtual glm::mat3 getMat3(int uniformId) const = 0;
        virtual glm::mat4 getMat4(int uniformId) const = 0;

        static ShaderProgram *create();
	};
}

#endif