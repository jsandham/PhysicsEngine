#ifndef VIEWPORT_H__
#define VIEWPORT_H__

namespace PhysicsEngine
{
    class Viewport
    {
        public:
            int mX;
            int mY;
            int mWidth;
            int mHeight;

        public:
            Viewport();
            Viewport(int x, int y, int width, int height);
            ~Viewport();
    };
}

#endif