#ifndef RAYTRACER_H__
#define RAYTRACER_H__

#include <vector>

#include "../core/glm.h"

namespace PhysicsEngine
{
	class World;
	class Camera;
	class Transform;

	class Raytracer
	{
      private:
        World *mWorld;

        //glm::vec3 mEye;
        //glm::vec3 mCentre;
        //glm::vec3 mUp;

        int mSamplesPerRay;
        std::vector<float> mImage;
        //std::vector<unsigned char> mFinalImage;

      public:
        Raytracer();
        ~Raytracer();
        Raytracer(const Raytracer &other) = delete;
        Raytracer &operator=(const Raytracer &other) = delete;

        void init(World *world);
        void update(Camera *camera);

	};
}

#endif