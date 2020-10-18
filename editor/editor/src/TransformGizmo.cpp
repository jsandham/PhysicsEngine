#include "core/ClosestDistance.h"
#include "core/Intersect.h"
#include "graphics/Graphics.h"

#include "../include/TransformGizmo.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

void TransformGizmo::initialize()
{
    mVertexShader = "#version 430 core\n"
                    "uniform mat4 mvp;\n"
                    "in vec3 position;\n"
                    "void main()\n"
                    "{\n"
                    "	gl_Position = mvp * vec4(position, 1.0);\n"
                    "}";

    mFragmentShader = "#version 430 core\n"
                      "uniform vec4 color;\n"
                      "out vec4 FragColor;\n"
                      "void main()\n"
                      "{\n"
                      "	FragColor = color;\n"
                      "}";

    Graphics::compile(mVertexShader, mFragmentShader, "", &mGizmoShaderProgram);
    mGizmoShaderMVPLoc = Graphics::findUniformLocation("mvp", mGizmoShaderProgram);
    mGizmoShaderColorLoc = Graphics::findUniformLocation("color", mGizmoShaderProgram);

    for (int i = 0; i < 3; i++)
    {
        float a = 1.0f * (i == 0);
        float b = 1.0f * (i == 1);
        float c = 1.0f * (i == 2);

        GLfloat translationVertices[] = {
            0.0f, 0.0f, 0.0f, // first vertex
            a,    b,    c     // second vertex
        };

        glGenVertexArrays(1, &mTranslationVAO[i]);
        glBindVertexArray(mTranslationVAO[i]);

        glGenBuffers(1, &mTranslationVBO[i]);
        glBindBuffer(GL_ARRAY_BUFFER, mTranslationVBO[i]);
        glBufferData(GL_ARRAY_BUFFER, sizeof(translationVertices), &translationVertices[0], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        Graphics::checkError();

        highlightedTransformAxis = Axis::Axis_None;
        selectedTransformAxis = Axis::Axis_None;
    }

    GLfloat rotationVertices[3 * 3 * 120];
    for (int i = 0; i < 60; i++)
    {
        float s1 = glm::cos(2 * glm::pi<float>() * i / 60.0f);
        float t1 = glm::sin(2 * glm::pi<float>() * i / 60.0f);
        float s2 = glm::cos(2 * glm::pi<float>() * (i + 1) / 60.0f);
        float t2 = glm::sin(2 * glm::pi<float>() * (i + 1) / 60.0f);

        rotationVertices[6 * i + 0] = 0.0f;
        rotationVertices[6 * i + 1] = s1;
        rotationVertices[6 * i + 2] = t1;
        rotationVertices[6 * i + 3] = 0.0f;
        rotationVertices[6 * i + 4] = s2;
        rotationVertices[6 * i + 5] = t2;

        rotationVertices[6 * i + 0 + 3 * 120] = s1;
        rotationVertices[6 * i + 1 + 3 * 120] = 0.0f;
        rotationVertices[6 * i + 2 + 3 * 120] = t1;
        rotationVertices[6 * i + 3 + 3 * 120] = s2;
        rotationVertices[6 * i + 4 + 3 * 120] = 0.0f;
        rotationVertices[6 * i + 5 + 3 * 120] = t2;

        rotationVertices[6 * i + 0 + 6 * 120] = s1;
        rotationVertices[6 * i + 1 + 6 * 120] = t1;
        rotationVertices[6 * i + 2 + 6 * 120] = 0.0f;
        rotationVertices[6 * i + 3 + 6 * 120] = s2;
        rotationVertices[6 * i + 4 + 6 * 120] = t2;
        rotationVertices[6 * i + 5 + 6 * 120] = 0.0f;
    }

    for (int i = 0; i < 3; i++)
    {
        glGenVertexArrays(1, &mRotationVAO[i]);
        glBindVertexArray(mRotationVAO[i]);

        glGenBuffers(1, &mRotationVBO[i]);
        glBindBuffer(GL_ARRAY_BUFFER, mRotationVBO[i]);
        glBufferData(GL_ARRAY_BUFFER, 3 * 120 * sizeof(GLfloat), &rotationVertices[i * 3 * 120], GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (void *)0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    glEnable(GL_LINE_SMOOTH);

    Graphics::checkError();

    mode = GizmoMode::Translation;
}

void TransformGizmo::update(PhysicsEngine::EditorCameraSystem *cameraSystem, PhysicsEngine::GizmoSystem* gizmoSystem,
                            PhysicsEngine::Transform *selectedTransform, float contentWidth, float contentHeight)
{
    assert(cameraSystem != NULL);
    assert(selectedTransform != NULL);

    switch (mode)
    {
    case GizmoMode::Translation:
        updateTranslation(cameraSystem, selectedTransform, contentWidth, contentHeight);
        break;
    case GizmoMode::Rotation:
        updateRotation(cameraSystem, gizmoSystem, selectedTransform, contentWidth, contentHeight);
        break;
    case GizmoMode::Scale:
        updateScale(cameraSystem, selectedTransform, contentWidth, contentHeight);
        break;
    }
}

void TransformGizmo::setGizmoMode(GizmoMode mode)
{
    this->mode = mode;
}

bool TransformGizmo::isGizmoHighlighted() const
{
    return highlightedTransformAxis != Axis::Axis_None;
}

void TransformGizmo::updateTranslation(PhysicsEngine::EditorCameraSystem *cameraSystem,
                                       PhysicsEngine::Transform *selectedTransform, float contentWidth,
                                       float contentHeight)
{
    glm::mat4 model = selectedTransform->getModelMatrix();
    glm::vec3 position = glm::vec3(model[3]);

    glm::vec3 xnormal = glm::normalize(glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
    glm::vec3 ynormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
    glm::vec3 znormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));

    Ray xAxisRay(position, xnormal);
    Ray yAxisRay(position, ynormal);
    Ray zAxisRay(position, znormal);

    float ndcX = 2 * (cameraSystem->getMousePosX() - 0.5f * contentWidth) / contentWidth;
    float ndcY = 2 * (cameraSystem->getMousePosY() - 0.5f * contentHeight) / contentHeight;

    Ray cameraRay = cameraSystem->normalizedDeviceSpaceToRay(ndcX, ndcY);

    float t = 0.0f;
    float xt = 0.0f;
    float yt = 0.0f;
    float zt = 0.0f;

    float closestDistanceToAxisX = ClosestDistance::closestDistance(cameraRay, xAxisRay, t, xt);
    float closestDistanceToAxisY = ClosestDistance::closestDistance(cameraRay, yAxisRay, t, yt);
    float closestDistanceToAxisZ = ClosestDistance::closestDistance(cameraRay, zAxisRay, t, zt);

    highlightedTransformAxis = Axis::Axis_None;
    if (xt >= 0.0f && xt <= 1.0f && closestDistanceToAxisX < closestDistanceToAxisY && closestDistanceToAxisX < closestDistanceToAxisZ)
    {
        highlightedTransformAxis = closestDistanceToAxisX < 0.05f ? Axis::Axis_X : highlightedTransformAxis;
    }
    else if (yt >= 0.0f && yt <= 1.0f && closestDistanceToAxisY < closestDistanceToAxisX && closestDistanceToAxisY < closestDistanceToAxisZ)
    {
        highlightedTransformAxis = closestDistanceToAxisY < 0.05f ? Axis::Axis_Y : highlightedTransformAxis;
    }
    else if (zt >= 0.0f && zt <= 1.0f && closestDistanceToAxisZ < closestDistanceToAxisX && closestDistanceToAxisZ < closestDistanceToAxisY)
    {
        highlightedTransformAxis = closestDistanceToAxisZ < 0.05f ? Axis::Axis_Z : highlightedTransformAxis;
    }

    if (cameraSystem->isLeftMouseHeldDown())
    {
        if (selectedTransformAxis == Axis::Axis_None)
        {
            selectedTransformAxis = highlightedTransformAxis;

            if (selectedTransformAxis == Axis::Axis_X)
            {
                start = xAxisRay.getPoint(xt);
            }
            else if (selectedTransformAxis == Axis::Axis_Y)
            {
                start = yAxisRay.getPoint(yt);
            }
            else if (selectedTransformAxis == Axis::Axis_Z)
            {
                start = zAxisRay.getPoint(zt);
            }
        }

        glm::vec3 delta = glm::vec3(0.0f, 0.0f, 0.0f);
        if (selectedTransformAxis == Axis::Axis_X)
        {
            end = xAxisRay.getPoint(xt);
            delta = end - start;
            start = end;
        }
        else if (selectedTransformAxis == Axis::Axis_Y)
        {
            end = yAxisRay.getPoint(yt);
            delta = end - start;
            start = end;
        }
        else if (selectedTransformAxis == Axis::Axis_Z)
        {
            end = zAxisRay.getPoint(zt);
            delta = end - start;
            start = end;
        }

        selectedTransform->mPosition += delta;
    }
    else {
        selectedTransformAxis = Axis::Axis_None;
    }

    drawTranslation(cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(), selectedTransform->getModelMatrix(),
                    cameraSystem->getNativeGraphicsMainFBO(), highlightedTransformAxis, selectedTransformAxis);
}

void TransformGizmo::updateRotation(PhysicsEngine::EditorCameraSystem *cameraSystem,
                                    PhysicsEngine::GizmoSystem *gizmoSystem,
                                    PhysicsEngine::Transform *selectedTransform, float contentWidth,
                                    float contentHeight)
{
    glm::mat4 model = selectedTransform->getModelMatrix();
    glm::vec3 position = glm::vec3(model[3]);

    glm::vec3 xnormal = glm::normalize(glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
    glm::vec3 ynormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
    glm::vec3 znormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));

    Circle xAxisCircle(position, xnormal, 1.0f);
    Circle yAxisCircle(position, ynormal, 1.0f);
    Circle zAxisCircle(position, znormal, 1.0f);

    float ndcX = 2 * (cameraSystem->getMousePosX() - 0.5f * contentWidth) / contentWidth;
    float ndcY = 2 * (cameraSystem->getMousePosY() - 0.5f * contentHeight) / contentHeight;

    Ray cameraRay = cameraSystem->normalizedDeviceSpaceToRay(ndcX, ndcY);

    float t = 0.0f;
    glm::vec3 xcirclePoint;
    glm::vec3 ycirclePoint;
    glm::vec3 zcirclePoint;

    float closestDistanceToAxisX = ClosestDistance::closestDistance(cameraRay, xAxisCircle, t, xcirclePoint);
    float closestDistanceToAxisY = ClosestDistance::closestDistance(cameraRay, yAxisCircle, t, ycirclePoint);
    float closestDistanceToAxisZ = ClosestDistance::closestDistance(cameraRay, zAxisCircle, t, zcirclePoint);

    highlightedTransformAxis = Axis::Axis_None;
    if (closestDistanceToAxisX < closestDistanceToAxisY && closestDistanceToAxisX < closestDistanceToAxisZ)
    {
        highlightedTransformAxis = closestDistanceToAxisX < 0.05f ? Axis::Axis_X : highlightedTransformAxis;
    }
    else if (closestDistanceToAxisY < closestDistanceToAxisX && closestDistanceToAxisY < closestDistanceToAxisZ)
    {
        highlightedTransformAxis = closestDistanceToAxisY < 0.05f ? Axis::Axis_Y : highlightedTransformAxis;
    }
    else if (closestDistanceToAxisZ < closestDistanceToAxisX && closestDistanceToAxisZ < closestDistanceToAxisY)
    {
        highlightedTransformAxis = closestDistanceToAxisZ < 0.05f ? Axis::Axis_Z : highlightedTransformAxis;
    }

    if (cameraSystem->isLeftMouseHeldDown())
    {
        if (selectedTransformAxis == Axis::Axis_None)
        {
            selectedTransformAxis = highlightedTransformAxis;

            if (selectedTransformAxis == Axis::Axis_X)
            {
                start = glm::normalize(xcirclePoint - xAxisCircle.mCentre);
                normal = xAxisCircle.mNormal;
            }
            else if (selectedTransformAxis == Axis::Axis_Y)
            {
                start = glm::normalize(ycirclePoint - yAxisCircle.mCentre);
                normal = yAxisCircle.mNormal;
            }
            else if (selectedTransformAxis == Axis::Axis_Z)
            {
                start = glm::normalize(zcirclePoint - zAxisCircle.mCentre);
                normal = zAxisCircle.mNormal;
            }
        }

        float angle = 0.0f;
        float sign = 1.0f;
        if (selectedTransformAxis == Axis::Axis_X)
        {
            end = glm::normalize(xcirclePoint - xAxisCircle.mCentre);
            angle = glm::acos(glm::clamp(glm::dot(start, end), -1.0f, 1.0f));
            sign = glm::sign(glm::dot(glm::cross(start, end), normal));
            start = end;

            gizmoSystem->addToDrawList(Line(xAxisCircle.mCentre, xAxisCircle.mCentre + 2.0f * normal), Color::cyan);
            gizmoSystem->addToDrawList(Line(xAxisCircle.mCentre, xcirclePoint), Color::magenta);
        }
        else if (selectedTransformAxis == Axis::Axis_Y)
        {
            end = glm::normalize(ycirclePoint - yAxisCircle.mCentre);
            angle = glm::acos(glm::clamp(glm::dot(start, end), -1.0f, 1.0f));
            sign = glm::sign(glm::dot(glm::cross(start, end), normal));
            start = end;

            gizmoSystem->addToDrawList(Line(yAxisCircle.mCentre, yAxisCircle.mCentre + 2.0f * normal), Color::cyan);
            gizmoSystem->addToDrawList(Line(yAxisCircle.mCentre, ycirclePoint), Color::magenta);
        }
        else if (selectedTransformAxis == Axis::Axis_Z)
        {
            end = glm::normalize(zcirclePoint - zAxisCircle.mCentre);
            angle = glm::acos(glm::clamp(glm::dot(start, end), -1.0f, 1.0f));
            sign = glm::sign(glm::dot(glm::cross(start, end), normal));
            start = end;

            gizmoSystem->addToDrawList(Line(zAxisCircle.mCentre, zAxisCircle.mCentre + 2.0f * normal), Color::cyan);
            gizmoSystem->addToDrawList(Line(zAxisCircle.mCentre, zcirclePoint), Color::magenta);
        }

        selectedTransform->mRotation = glm::angleAxis(sign * angle, normal) * selectedTransform->mRotation;
    }
    else {
        selectedTransformAxis = Axis::Axis_None;
    }

    drawRotation(cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(), selectedTransform->getModelMatrix(),
                 cameraSystem->getNativeGraphicsMainFBO(), highlightedTransformAxis, selectedTransformAxis);
}

void TransformGizmo::updateScale(PhysicsEngine::EditorCameraSystem *cameraSystem,
                                 PhysicsEngine::Transform *selectedTransform, float contentWidth, float contentHeight)
{
    drawScale(cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(), selectedTransform->getModelMatrix(),
              cameraSystem->getNativeGraphicsMainFBO(), highlightedTransformAxis, selectedTransformAxis);

    /*Plane plane0(selectedTransform->getUp(), selectedTransform->mPosition);
    Plane plane1(selectedTransform->getForward(), selectedTransform->mPosition);
    Plane plane2(selectedTransform->getRight(), selectedTransform->mPosition);

    float t0 = std::numeric_limits<float>().max();
    float t1 = std::numeric_limits<float>().max();
    float t2 = std::numeric_limits<float>().max();
    bool int0 = Intersect::intersect(cameraRay, plane0, t0);
    bool int1 = Intersect::intersect(cameraRay, plane1, t1);
    bool int2 = Intersect::intersect(cameraRay, plane2, t2);

    t0 = t0 < 0.0f ? std::numeric_limits<float>().max() : t0;
    t1 = t1 < 0.0f ? std::numeric_limits<float>().max() : t1;
    t2 = t2 < 0.0f ? std::numeric_limits<float>().max() : t2;

    if (int0 && t0 < t1 && t0 < t2) {
        gizmoSystem->addToDrawList(plane0, glm::vec3(1.0, 1.0, 1.0), Color::green);
    }
    else if(int1 && t1 < t0 && t1 < t2)
    {
        gizmoSystem->addToDrawList(plane1, glm::vec3(1.0, 1.0, 1.0), Color::blue);
    }
    else if(int2 && t2 < t0 && t2 < t1) {
        gizmoSystem->addToDrawList(plane2, glm::vec3(1.0, 1.0, 1.0), Color::red);
    }*/
}

void TransformGizmo::drawTranslation(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo,
                                     Axis highlightAxis, Axis selectedAxis)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClear(GL_DEPTH_BUFFER_BIT);

    for (int i = 0; i < 3; i++)
    {
        glm::vec4 axis_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        axis_color[i] = 1.0f;

        if (i == static_cast<int>(highlightAxis))
        {
            axis_color = glm::vec4(1.0f, 0.65f, 0.0f, 1.0f);
        }

        if (selectedAxis != Axis::Axis_None)
        {
            if (i != static_cast<int>(selectedAxis))
            {
                axis_color = glm::vec4(0.65f, 0.65f, 0.65f, 1.0f);
            }
            else
            {
                axis_color = glm::vec4(1.0f, 0.65f, 0.0f, 1.0f);
            }
        }

        glm::mat4 mvp = projection * view * model;

        Graphics::use(mGizmoShaderProgram);
        Graphics::setMat4(mGizmoShaderMVPLoc, mvp);
        Graphics::setVec4(mGizmoShaderColorLoc, axis_color);

        glBindVertexArray(mTranslationVAO[i]);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void TransformGizmo::drawRotation(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, Axis highlightAxis,
                                  Axis selectedAxis)
{
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClear(GL_DEPTH_BUFFER_BIT);

    for (int i = 0; i < 3; i++)
    {
        glm::vec4 axis_color = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        axis_color[i] = 1.0f;

        if (i == static_cast<int>(highlightAxis))
        {
            axis_color = glm::vec4(1.0f, 0.65f, 0.0f, 1.0f);
        }

        if (selectedAxis != Axis::Axis_None)
        {
            if (i != static_cast<int>(selectedAxis))
            {
                axis_color = glm::vec4(0.65f, 0.65f, 0.65f, 1.0f);
            }
            else
            {
                axis_color = glm::vec4(1.0f, 0.65f, 0.0f, 1.0f);
            }
        }

        glm::mat4 mvp = projection * view * model;

        Graphics::use(mGizmoShaderProgram);
        Graphics::setMat4(mGizmoShaderMVPLoc, mvp);
        Graphics::setVec4(mGizmoShaderColorLoc, axis_color);

        glBindVertexArray(mRotationVAO[i]);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 120);
        glBindVertexArray(0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void TransformGizmo::drawScale(glm::mat4 projection, glm::mat4 view, glm::mat4 model, GLuint fbo, Axis highlightAxis,
                               Axis selectedAxis)
{
}