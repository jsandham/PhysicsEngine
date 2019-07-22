#ifndef __INSPECTOR_H__
#define __INSPECTOR_H__

#include "core\Entity.h"

using namespace PhysicsEngine;

namespace PhysicsEditor
{
	class Inspector
	{
		public:
			Inspector();
			~Inspector();

			void render(Entity* entity, bool isOpenedThisFrame);





			bool BeginAddComponentDropdown(std::string filter, std::string& componentToAdd);
			void EndAddComponentDropdown();
	};
}

#endif