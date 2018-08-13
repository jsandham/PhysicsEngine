#ifndef __GRAPHICS_H__
#define __GRAPHICS_H__

#include "../core/Texture2D.h"
#include "../core/Texture3D.h"
#include "../core/Shader.h"

namespace PhysicsEngine
{
	class Graphics
	{
		public:
			static void initializeGraphicsAPI();

			static void readPixels(Texture2D* texture);
			static void apply(Texture2D* texture);
			static void generate(Texture2D* texture);
			static void destroy(Texture2D* texture);
			static void bind(Texture2D* texture);
			static void unbind(Texture2D* texture);
			static void active(Texture2D* texture, unsigned int slot);

			static void readPixels(Texture3D* texture);
			static void apply(Texture3D* texture);
			static void generate(Texture3D* texture);
			static void destroy(Texture3D* texture);
			static void bind(Texture3D* texture);
			static void unbind(Texture3D* texture);
			static void active(Texture3D* texture, unsigned int slot);

			static void compile(Shader* shader);
			static void bind(Shader* shader);
			static void unbind(Shader* shader);
	};
}

#endif