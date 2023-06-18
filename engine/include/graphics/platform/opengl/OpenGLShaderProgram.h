#ifndef OPENGLSHADERPROGRAM_H__
#define OPENGLSHADERPROGRAM_H__

#include "../../../../include/graphics/ShaderProgram.h"

namespace PhysicsEngine
{
class OpenGLShaderProgram : public ShaderProgram
{
    unsigned int mHandle;

  public:
    OpenGLShaderProgram();
    ~OpenGLShaderProgram();

    void load(const std::string& name, const std::string &vertex, const std::string &fragment, const std::string &geometry) override;
    void load(const std::string& name, const std::string &vertex, const std::string &fragment) override;
    void compile() override;
    void bind() override;
    void unbind() override;
    int findUniformLocation(const std::string &name) const override;

    std::vector<ShaderUniform> getUniforms() const override;
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

    void setBool(int nameLocation, bool value) override;
    void setInt(int nameLocation, int value) override;
    void setFloat(int nameLocation, float value) override;
    void setColor(int nameLocation, const Color &color) override;
    void setColor32(int nameLocation, const Color32 &color) override;
    void setVec2(int nameLocation, const glm::vec2 &vec) override;
    void setVec3(int nameLocation, const glm::vec3 &vec) override;
    void setVec4(int nameLocation, const glm::vec4 &vec) override;
    void setMat2(int nameLocation, const glm::mat2 &mat) override;
    void setMat3(int nameLocation, const glm::mat3 &mat) override;
    void setMat4(int nameLocation, const glm::mat4 &mat) override;
    void setTexture2D(int nameLocation, int texUnit, void* tex) override;
    void setTexture2Ds(int nameLocation, const std::vector<int>& texUnits, int count, const std::vector<void*>& texs) override;

    bool getBool(const char *name) const override;
    int getInt(const char *name) const override;
    float getFloat(const char *name) const override;
    Color getColor(const char *name) const override;
    glm::vec2 getVec2(const char *name) const override;
    glm::vec3 getVec3(const char *name) const override;
    glm::vec4 getVec4(const char *name) const override;
    glm::mat2 getMat2(const char *name) const override;
    glm::mat3 getMat3(const char *name) const override;
    glm::mat4 getMat4(const char *name) const override;

    bool getBool(int nameLocation) const override;
    int getInt(int nameLocation) const override;
    float getFloat(int nameLocation) const override;
    Color getColor(int nameLocation) const override;
    Color32 getColor32(int nameLocation) const override;
    glm::vec2 getVec2(int nameLocation) const override;
    glm::vec3 getVec3(int nameLocation) const override;
    glm::vec4 getVec4(int nameLocation) const override;
    glm::mat2 getMat2(int nameLocation) const override;
    glm::mat3 getMat3(int nameLocation) const override;
    glm::mat4 getMat4(int nameLocation) const override;
};
} // namespace PhysicsEngine

#endif