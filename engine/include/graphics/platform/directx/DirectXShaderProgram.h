#ifndef DIRECTXSHADERPROGRAM_H__
#define DIRECTXSHADERPROGRAM_H__

#include "../../../../include/graphics/ShaderProgram.h"
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>

namespace PhysicsEngine
{
class DirectXShaderProgram : public ShaderProgram
{
  private:
    ID3D11VertexShader *mVertexShader;
    ID3D11PixelShader *mPixelShader;
    ID3D11GeometryShader *mGeometryShader;

    ID3DBlob *mVertexShaderBlob;
    ID3DBlob *mPixelShaderBlob;
    ID3DBlob *mGeometryShaderBlob;

    D3D11_BUFFER_DESC mVSConstantBufferDesc;
    D3D11_BUFFER_DESC mPSConstantBufferDesc;
    ID3D11Buffer *mVSConstantBuffer;
    ID3D11Buffer *mPSConstantBuffer;

  public:
    DirectXShaderProgram();
    ~DirectXShaderProgram();

    void load(const std::string& name, const std::string &vertex, const std::string &fragment, const std::string &geometry) override;
    void load(const std::string& name, const std::string &vertex, const std::string &fragment) override;
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
    void setTexture2D(const char *name, int texUnit, void* tex) override;
    void setTexture2Ds(const char *name, const std::vector<int>& texUnits, int count, const std::vector<void*>& texs) override;

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
    Color32 getColor32(const char* name) const override;
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
};
} // namespace PhysicsEngine

#endif