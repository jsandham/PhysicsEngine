#ifndef __MATERIAL_RENDERER_H__
#define __MATERIAL_RENDERER_H__

#include <GL/glew.h>
#include <gl/gl.h>

#include "core/Mesh.h"
#include "core/Shader.h"
#include "core/Material.h"

#define GLM_FORCE_RADIANS

#include "../glm/glm.hpp"
#include "../glm/gtx/quaternion.hpp"
#include "../glm/gtc/matrix_transform.hpp"

namespace PhysicsEditor
{
	class MaterialRenderer
	{
		private:
			GLuint fbo;
			GLuint colorTex;
			GLuint depthTex;

			PhysicsEngine::Mesh mesh;
			//PhysicsEngine::Shader shader;

			glm::mat4 model;
			glm::mat4 view;
			glm::mat4 projection;
			glm::vec3 cameraPos;

		public:
			MaterialRenderer();
			~MaterialRenderer();

			void init();
			void render(PhysicsEngine::World* world, PhysicsEngine::Material* material);

			GLuint getColorTarget() const;
	};
}

#endif
