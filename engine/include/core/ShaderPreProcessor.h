#ifndef SHADER_PREPROCESSOR_H__
#define SHADER_PREPROCESSOR_H__

#include <string>
#include <vector>

namespace PhysicsEngine
{
std::vector<int> computeShaderVariants(const std::string &vert, const std::string &frag, const std::string &geom);
std::string buildShaderVariant(int variant);
} // namespace PhysicsEngine

#endif