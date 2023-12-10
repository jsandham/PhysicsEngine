#ifndef DIRECTXSHADERPROGRAM_H__
#define DIRECTXSHADERPROGRAM_H__

#define NOMINMAX
#include <d3d11.h>
#include <d3d11shader.h>
#include <windows.h>

#include "../../ShaderProgram.h"
#include "../../UniformBuffer.h"

namespace PhysicsEngine
{
struct ConstantBufferVariable
{
    PipelineStage mStage;
    int mUniformId;
    unsigned int mConstantBufferIndex;
    unsigned int mSize;
    unsigned int mOffset;
};

// struct ShaderUniform
//{
//     char mData[64];
//     std::string mName;       // variable name (including block name if applicable)
//     ShaderUniformType mType; // type of the uniform (float, vec3 or mat4, etc)
//     void *mTex;              // if data stores a texture id, this is the texture handle
//     int mUniformId;          // integer hash of uniform name

//    std::string getShortName() const
//    {
//        size_t pos = mName.find_first_of('.');
//        return mName.substr(pos + 1);
//    }
//};

class DirectXShaderProgram : public ShaderProgram
{
  private:
    ID3D11VertexShader *mVertexShader;
    ID3D11PixelShader *mPixelShader;
    ID3D11GeometryShader *mGeometryShader;

    ID3DBlob *mVertexShaderBlob;
    ID3DBlob *mPixelShaderBlob;
    ID3DBlob *mGeometryShaderBlob;

    std::vector<UniformBuffer *> mVSConstantBuffers;
    std::vector<UniformBuffer *> mPSConstantBuffers;
    std::vector<UniformBuffer *> mGSConstantBuffers;

    std::vector<ConstantBufferVariable> mConstantBufferVariables;

  public:
    DirectXShaderProgram();
    ~DirectXShaderProgram();

    ID3DBlob *getVSBlob() const;
    ID3DBlob *getPSBlob() const;
    ID3DBlob *getGSBlob() const;

    void load(const std::string &name, const std::string &vertex, const std::string &fragment,
              const std::string &geometry) override;
    void load(const std::string &name, const std::string &vertex, const std::string &fragment) override;
    void compile() override;
    void bind() override;
    void unbind() override;

    std::vector<ShaderUniform> getUniforms() const override;
    std::vector<ShaderUniform> getMaterialUniforms() const override;
    std::vector<ShaderAttribute> getAttributes() const override;

    void setBool(const char *name, bool value) override;
    void setInt(const char *name, int value) override;
    void setFloat(const char *name, float value) override;
    void setColor(const char *name, const Color &color) override;
    void setColor32(const char *name, const Color32 &color) override;
    void setVec2(const char *name, const glm::vec2 &vec) override;
    void setVec3(const char *name, const glm::vec3 &vec) override;
    void setVec4(const char *name, const glm::vec4 &vec) override;
    void setMat2(const char *name, const glm::mat2 &mat) override;
    void setMat3(const char *name, const glm::mat3 &mat) override;
    void setMat4(const char *name, const glm::mat4 &mat) override;
    void setTexture2D(const char *name, int texUnit, void *tex) override;
    void setTexture2Ds(const char *name, const std::vector<int> &texUnits, int count,
                       const std::vector<void *> &texs) override;

    void setBool(int uniformId, bool value) override;
    void setInt(int uniformId, int value) override;
    void setFloat(int uniformId, float value) override;
    void setColor(int uniformId, const Color &color) override;
    void setColor32(int uniformId, const Color32 &color) override;
    void setVec2(int uniformId, const glm::vec2 &vec) override;
    void setVec3(int uniformId, const glm::vec3 &vec) override;
    void setVec4(int uniformId, const glm::vec4 &vec) override;
    void setMat2(int uniformId, const glm::mat2 &mat) override;
    void setMat3(int uniformId, const glm::mat3 &mat) override;
    void setMat4(int uniformId, const glm::mat4 &mat) override;
    void setTexture2D(int uniformId, int texUnit, void *tex) override;
    void setTexture2Ds(int uniformId, const std::vector<int> &texUnits, int count,
                       const std::vector<void *> &texs) override;

    bool getBool(const char *name) const override;
    int getInt(const char *name) const override;
    float getFloat(const char *name) const override;
    Color getColor(const char *name) const override;
    Color32 getColor32(const char *name) const override;
    glm::vec2 getVec2(const char *name) const override;
    glm::vec3 getVec3(const char *name) const override;
    glm::vec4 getVec4(const char *name) const override;
    glm::mat2 getMat2(const char *name) const override;
    glm::mat3 getMat3(const char *name) const override;
    glm::mat4 getMat4(const char *name) const override;

    bool getBool(int uniformId) const override;
    int getInt(int uniformId) const override;
    float getFloat(int uniformId) const override;
    Color getColor(int uniformId) const override;
    Color32 getColor32(int uniformId) const override;
    glm::vec2 getVec2(int uniformId) const override;
    glm::vec3 getVec3(int uniformId) const override;
    glm::vec4 getVec4(int uniformId) const override;
    glm::mat2 getMat2(int uniformId) const override;
    glm::mat3 getMat3(int uniformId) const override;
    glm::mat4 getMat4(int uniformId) const override;

  private:
    void setData(int uniformId, const void *data);
    void getData(int uniformId, void *data, size_t sizeInBytes) const;
};
} // namespace PhysicsEngine

#endif