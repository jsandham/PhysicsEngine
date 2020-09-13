#ifndef __CIRCLE_H__
#define __CIRCLE_H__

#include "../glm/glm.hpp"
#include "../glm/gtc/constants.hpp"

namespace PhysicsEngine
{
	class Circle
	{
	public:
		glm::vec3 mCentre;
		glm::vec3 mNormal;
		float mRadius;

	public:
		Circle();
		Circle(glm::vec3 centre, glm::vec3 normal, float radius);
		~Circle();

		float getArea() const;
		float getCircumference() const;
	};
}

#endif