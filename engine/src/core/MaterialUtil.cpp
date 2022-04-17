//#include <vector>
//
//#include "../../include/core/Log.h"
//#include "../../include/core/Material.h"
//#include "../../include/core/MaterialUtil.h"
//#include "../../include/core/World.h"
//
//using namespace PhysicsEngine;
//
//void MaterialUtil::copyMaterialTo(World *srcWorld, Material *srcMat, World *destWorld, Material *destMat)
//{
//    Guid shaderId = srcMat->getShaderId();
//
//    if (destWorld->getAssetById<Shader>(shaderId) == NULL)
//    {
//        std::string message = "Shader with id: " + shaderId.toString() + " does not exist in destination world\n";
//        Log::error(message.c_str());
//        return;
//    }
//
//    destMat->setShaderId(shaderId);
//
//    // Copy uniforms from source material to destination material
//    std::vector<ShaderUniform> uniforms = srcMat->getUniforms();
//    for (size_t i = 0; i < uniforms.size(); i++)
//    {
//
//        // Note: matrices not supported
//        switch (uniforms[i].mType)
//        {
//        case ShaderUniformType::Int/*GL_INT*/:
//            destMat->setInt(uniforms[i].mName, srcMat->getInt(uniforms[i].mName));
//            break;
//        case ShaderUniformType::Float/*GL_FLOAT*/:
//            destMat->setFloat(uniforms[i].mName, srcMat->getFloat(uniforms[i].mName));
//            break;
//        case ShaderUniformType::Color /*GL_FLOAT*/:
//            destMat->setColor(uniforms[i].mName, srcMat->getColor(uniforms[i].mName));
//            break;
//        case ShaderUniformType::Vec2/*GL_FLOAT_VEC2*/:
//            destMat->setVec2(uniforms[i].mName, srcMat->getVec2(uniforms[i].mName));
//            break;
//        case ShaderUniformType::Vec3/*GL_FLOAT_VEC3*/:
//            destMat->setVec3(uniforms[i].mName, srcMat->getVec3(uniforms[i].mName));
//            break;
//        case ShaderUniformType::Vec4/*GL_FLOAT_VEC4*/:
//            destMat->setVec4(uniforms[i].mName, srcMat->getVec4(uniforms[i].mName));
//            break;
//        case ShaderUniformType::Sampler2D/*GL_SAMPLER_2D*/:
//            destMat->setTexture(uniforms[i].mName, srcMat->getTexture(uniforms[i].mName));
//            break;
//        }
//    }
//}