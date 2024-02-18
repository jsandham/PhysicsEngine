#include "../../include/core/Rect.h"

using namespace PhysicsEngine;

Rect::Rect() : mX(0.0f), mY(0.0f), mWidth(1.0f), mHeight(1.0f)
{
}

Rect::Rect(float x, float y, float width, float height) : mX(x), mY(y), mWidth(width), mHeight(height)
{
}

Rect::Rect(const glm::vec2 &min, const glm::vec2 &max)
{
    mX = min.x;
    mY = min.y;
    mWidth = max.x - min.x;
    mHeight = max.y - min.y;
}

bool Rect::contains(float x, float y) const
{
    float maxX = mX + mWidth;
    float maxY = mY + mHeight;
    return x >= mX && y >= mY && x <= maxX && y <= maxY;
}

float Rect::getArea() const
{
    return mWidth * mHeight;
}

glm::vec2 Rect::getCentre() const
{
    return glm::vec2(mX + 0.5f * mWidth, mY + 0.5f * mHeight);
}

glm::vec2 Rect::getMin() const
{
    return glm::vec2(mX, mY);
}

glm::vec2 Rect::getMax() const
{
    return glm::vec2(mX + mWidth, mY + mHeight);
}