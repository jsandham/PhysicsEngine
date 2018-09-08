#ifndef __MATERIAL_H__
#define __MATERIAL_H__

namespace PhysicsEngine
{
	typedef enum TEXURESLOT
	{
		MAINTEXTURE,
		NORMAL,
		DIFFUSE,
		SPECULAR,
		ALBEDO,
		GLOSS,
		CUBEMAP,
		COUNT
	}TEXTURESLOT;

	class Material
	{
		public:
			int materialId;
			int shaderId;
			int textureId;

			// int globalMaterialIndex;
			// int globalTextureIndex;
			// int globalShaderIndex;

		public:
			Material();
			~Material();
	};
}

#endif




















// #ifndef __MATERIAL_H__
// #define __MATERIAL_H__

// #include <string>

// #include "Texture2D.h"
// #include "Cubemap.h"

// #include "Shader.h"
// #include "ShaderUniformState.h"
// #include "Color.h"
// #include "GraphicState.h"

// #define GLM_FORCE_RADIANS

// #include "../glm/glm.hpp"

// namespace PhysicsEngine
// {
// 	class Material
// 	{
// 		private:
// 			std::vector<Texture2D*> textures;
// 			Shader *shader;
// 			ShaderUniformState *uniforms;

// 			Texture2D *mainTexture;
// 			Texture2D *normalMap;
// 			Texture2D *diffuseMap;
// 			Texture2D *specularMap;
// 			Texture2D *albedoMap;
// 			Texture2D *glossMap;
// 			Cubemap *cubemap;

// 			float shininess;
// 			glm::vec4 color;
// 			glm::vec3 ambient;
// 			glm::vec3 diffuse;
// 			glm::vec3 specular;

// 		public:
// 			Material(Shader* shader);
// 			Material(const Material &material);
// 			Material& operator=(const Material &material);
// 			~Material();

// 			void bind(GraphicState& state);
// 			void unbind();

// 			void setShininess(float shininess);
// 			void setColor(glm::vec4 &color);
// 			void setAmbient(glm::vec3 &ambient);
// 			void setDiffuse(glm::vec3 &diffuse);
// 			void setSpecular(glm::vec3 &specular);

// 			void setMainTexture(Texture2D *texture);
// 			void setNormalMap(Texture2D *texture);
// 			void setDiffuseMap(Texture2D *texture);
// 			void setSpecularMap(Texture2D *texture);
// 			void setAlbedoMap(Texture2D *texture);
// 			void setGlossMap(Texture2D *texture);
// 			void setCubemap(Cubemap *texture);

// 			void setBool(std::string name, bool value);
// 			void setInt(std::string name, int value);
// 			void setFloat(std::string name, float value);
// 			void setVec2(std::string name, glm::vec2 &vec);
// 			void setVec3(std::string name, glm::vec3 &vec);
// 			void setVec4(std::string name, glm::vec4 &vec);
// 			void setMat2(std::string name, glm::mat2 &mat);
// 			void setMat3(std::string name, glm::mat3 &mat);
// 			void setMat4(std::string name, glm::mat4 &mat);

// 			float getShininess();
// 			glm::vec4& getColor();
// 			glm::vec3& getAmbient();
// 			glm::vec3& getDiffuse();
// 			glm::vec3& getSpecular();

// 			Texture2D* getMainTexture();
// 			Texture2D* getNormalMap();
// 			Texture2D* getDiffuseMap();
// 			Texture2D* getSpecularMap();
// 			Texture2D* getAlbedoMap();
// 			Texture2D* getGlossMap();
// 			Cubemap* getCubemap();
	
// 			Shader* getShader();
// 	};
// }

// #endif