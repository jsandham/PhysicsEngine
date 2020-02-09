#ifndef __MATERIAL_DRAWER_H__
#define __MATERIAL_DRAWER_H__

#include "InspectorDrawer.h"

namespace PhysicsEditor
{
	class MaterialDrawer : public InspectorDrawer
	{
	public:
		MaterialDrawer();
		~MaterialDrawer();

		void render(World* world, EditorProject& project, EditorScene& scene, EditorClipboard& clipboard, Guid id);
	};



	//template <GLenum T>
	//struct UniformDrawer
	//{
	//	static void draw();
	//};

	//template<GLenum T>
	//void UniformDrawer<T>::draw() {

	//}

	//template<>
	//void UniformDrawer<GL_INT>::draw()
	//{

	//}

	//template<>
	//void UniformDrawer<GL_INT_VEC2>::draw()
	//{

	//}

	/*template<>
	void UniformDrawer<GL_INT_VEC3>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_INT_VEC4>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_FLOAT>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_FLOAT_VEC2>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_FLOAT_VEC3>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_FLOAT_VEC4>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_FLOAT_MAT2>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_FLOAT_MAT3>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_FLOAT_MAT4>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_SAMPLER_2D>::draw()
	{

	}

	template<>
	void UniformDrawer<GL_SAMPLER_CUBE>::draw()
	{

	}*/

	
}

#endif