// #ifndef __GRAPHICSTATE_H__
// #define __GRAPHICSTATE_H__

// #include <GL/glew.h>
// #include <gl/gl.h>

// #include "../glm/glm.hpp"
// #include "../glm/gtc/type_ptr.hpp"

// #include "Buffer.h"
// #include "../core/Texture2D.h"
// #include "../core/Cubemap.h"

// #include "../graphics/GLHandle.h"

// namespace PhysicsEngine
// {
// 	typedef enum UniformBuffer
// 	{
// 		CameraBuffer,
// 		ShadowBuffer,
// 		DirectionalLightBuffer,
// 		SpotLightBuffer,
// 		PointLightBuffer
// 	}UniformBuffer;

// 	class GraphicState
// 	{
// 		private:                          // byte offset for std140 layout
// 			glm::mat4 projection;         // 0
// 			glm::mat4 view;               // 64
// 			glm::vec3 cameraPos;          // 128

// 			glm::mat4 lightProjection[5]; // 0    64   128  192  256
// 			glm::mat4 lightView[5];       // 320  384  448  512  576 
// 			float cascadeEnd[5];          // 640  656  672  688  704
// 			float farPlane;               // 720

// 			glm::vec3 dirLightDirection;  // 0
// 			glm::vec3 dirLightAmbient;    // 16
// 			glm::vec3 dirLightDiffuse;    // 32
// 			glm::vec3 dirLightSpecular;   // 48

// 			glm::vec3 spotLightPosition;  // 0
// 			glm::vec3 spotLightDirection; // 16
// 			glm::vec3 spotLightAmbient;   // 32
// 			glm::vec3 spotLightDiffuse;   // 48
// 			glm::vec3 spotLightSpecular;  // 64
// 			float spotLightConstant;      // 80
// 			float spotLightLinear;        // 84
// 			float spotLightQuadratic;     // 88
// 			float spotLightCutoff;        // 92
// 			float spotLightOuterCutoff;   // 96

// 			glm::vec3 pointLightPosition; // 0
// 			glm::vec3 pointLightAmbient;  // 16
// 			glm::vec3 pointLightDiffuse;  // 32
// 			glm::vec3 pointLightSpecular; // 48
// 			float pointLightConstant;     // 64
// 			float pointLightLinear;       // 68
// 			float pointLightQuadratic;    // 72


// 			Buffer buffers[5];

// 		public:
// 			std::vector<Texture2D*> cascadeTexture2D;
// 			Texture2D* shadowTexture2D;
// 			Cubemap* shadowCubemap;

// 		public:
// 			GraphicState();
// 			~GraphicState();

// 			void init();
// 			void bind(UniformBuffer state);
// 			void unbind(UniformBuffer state);

// 			/*void setMat4(UniformState state, Uniform uniform, glm::mat4 mat);
// 			void setVec3(UniformState state, Uniform uniform, glm::vec3 vec);
// 			void setFloat(UniformState state, Uniform uniform, float scalar);
// 			void setInt(UniformState state, Uniform uniform, int scalar);

// 			glm::mat4 getMat4(UniformState state, Uniform uniform);
// 			glm::vec3 getVec3(UniformState state, Uniform uniform);
// 			float getFloat(UniformState state, Uniform uniform);
// 			int getInt(UniformState state, Uniform uniform);
// */

// 			void setProjectionMatrix(glm::mat4 projection);
// 			void setViewMatrix(glm::mat4 view);
// 			void setCameraPosition(glm::vec3 position);

// 			void setLightProjectionMatrix(glm::mat4 projection, int index);
// 			void setLightViewMatrix(glm::mat4 view, int index);
// 			void setCascadeEnd(float cascadeEnd, int index);
// 			void setFarPlane(float farPlane);
			
// 			void setDirLightDirection(glm::vec3 direction);
// 			void setDirLightAmbient(glm::vec3 ambient);
// 			void setDirLightDiffuse(glm::vec3 diffuse);
// 			void setDirLightSpecular(glm::vec3 specular);
			
// 			void setSpotLightDirection(glm::vec3 direction);
// 			void setSpotLightPosition(glm::vec3 position);
// 			void setSpotLightAmbient(glm::vec3 ambient);
// 			void setSpotLightDiffuse(glm::vec3 diffuse);
// 			void setSpotLightSpecular(glm::vec3 specular);

// 			void setPointLightPosition(glm::vec3 position);
// 			void setPointLightAmbient(glm::vec3 ambient);
// 			void setPointLightDiffuse(glm::vec3 diffuse);
// 			void setPointLightSpecular(glm::vec3 specular);

// 			void setSpotLightConstant(float constant);
// 			void setSpotLightLinear(float linear);
// 			void setSpotLightQuadratic(float quadratic);
// 			void setSpotLightCutoff(float cutoff);
// 			void setSpotLightOuterCutoff(float cutoff);
// 			void setPointLightConstant(float constant);
// 			void setPointLightLinear(float linear);
// 			void setPointLightQuadratic(float quadratic);


// 			glm::mat4 getProjectionMatrix();
// 			glm::mat4 getViewMatrix();
// 			glm::vec3 getCameraPosition();

// 			glm::mat4 getLightProjectionMatrix(int index);
// 			glm::mat4 getLightViewMatrix(int index);
// 			float getCascadeEnd(int index);
// 			float getFarPlane();

// 			glm::vec3 getDirLightDirection();
// 			glm::vec3 getDirLightAmbient();
// 			glm::vec3 getDirLightDiffuse();
// 			glm::vec3 getDirLightSpecular();

// 			glm::vec3 getSpotLightDirection();
// 			glm::vec3 getSpotLightPosition();
// 			glm::vec3 getSpotLightAmbient();
// 			glm::vec3 getSpotLightDiffuse();
// 			glm::vec3 getSpotLightSpecular();

// 			glm::vec3 getPointLightPosition();
// 			glm::vec3 getPointLightAmbient();
// 			glm::vec3 getPointLightDiffuse();
// 			glm::vec3 getPointLightSpecular();

// 			float getSpotLightConstant();
// 			float getSpotLightLinear();
// 			float getSpotLightQuadratic();
// 			float getSpotLightCutoff();
// 			float getSpotLightOuterCutoff();
// 			float getPointLightConstant();
// 			float getPointLightLinear();
// 			float getPointLightQuadratic();
// 	};
// }

// #endif