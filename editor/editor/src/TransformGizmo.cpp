#include "core/ClosestDistance.h"
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

void TransformGizmo::update(PhysicsEngine::EditorCameraSystem *cameraSystem,
                            PhysicsEngine::Transform *selectedTransform, float contentWidth, float contentHeight)
{
    assert(cameraSystem != NULL);
    assert(selectedTransform != NULL);

    if (mode == GizmoMode::Translation)
    {
        updateTranslation(cameraSystem, selectedTransform, contentWidth, contentHeight);
    }
    else if (mode == GizmoMode::Rotation)
    {
        updateRotation(cameraSystem, selectedTransform, contentWidth, contentHeight);
    }
    else
    {
        updateScale(cameraSystem, selectedTransform, contentWidth, contentHeight);
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
    if (!cameraSystem->isLeftMouseHeldDown())
    {
        selectedTransformAxis = Axis::Axis_None;
    }

    glm::mat4 model = selectedTransform->getModelMatrix();

    glm::vec3 position = glm::vec3(model[3]);

    Ray xAxisRay(position, glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
    Ray yAxisRay(position, glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
    Ray zAxisRay(position, glm::vec3(model * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));

    float width = contentWidth;   // (float)(sceneContentMax.x - sceneContentMin.x);
    float height = contentHeight; // (float)(sceneContentMax.y - sceneContentMin.y);
    float ndcX = 2 * (cameraSystem->getMousePosX() - 0.5f * width) / width;
    float ndcY = 2 * (cameraSystem->getMousePosY() - 0.5f * height) / height;

    Ray cameraRay = cameraSystem->normalizedDeviceSpaceToRay(ndcX, ndcY);

    float closestDistanceToAxisX = ClosestDistance::closestDistance(cameraRay, xAxisRay);
    float closestDistanceToAxisY = ClosestDistance::closestDistance(cameraRay, yAxisRay);
    float closestDistanceToAxisZ = ClosestDistance::closestDistance(cameraRay, zAxisRay);

    highlightedTransformAxis = Axis::Axis_None;
    if (closestDistanceToAxisX < closestDistanceToAxisY && closestDistanceToAxisX < closestDistanceToAxisZ)
    {
        if (closestDistanceToAxisX < 0.05f)
        {
            highlightedTransformAxis = Axis::Axis_X;
        }
    }
    else if (closestDistanceToAxisY < closestDistanceToAxisX && closestDistanceToAxisY < closestDistanceToAxisZ)
    {
        if (closestDistanceToAxisY < 0.05f)
        {
            highlightedTransformAxis = Axis::Axis_Y;
        }
    }
    else if (closestDistanceToAxisZ < closestDistanceToAxisX && closestDistanceToAxisZ < closestDistanceToAxisY)
    {
        if (closestDistanceToAxisZ < 0.05f)
        {
            highlightedTransformAxis = Axis::Axis_Z;
        }
    }

    if (cameraSystem->isLeftMouseHeldDown())
    {
        if (selectedTransformAxis == Axis::Axis_None)
        {
            selectedTransformAxis = highlightedTransformAxis;
            selectedTransformModel = model;
        }

        glm::vec2 delta = cameraSystem->distanceTraveledSinceLeftMouseClick();

        if (selectedTransformAxis == Axis::Axis_X)
        {
            selectedTransform->mPosition = glm::vec3(selectedTransformModel * glm::vec4(0.05f * delta.x, 0, 0, 1));
        }
        else if (selectedTransformAxis == Axis::Axis_Y)
        {
            selectedTransform->mPosition = glm::vec3(selectedTransformModel * glm::vec4(0, 0.05f * delta.x, 0, 1));
        }
        else if (selectedTransformAxis == Axis::Axis_Z)
        {
            selectedTransform->mPosition = glm::vec3(selectedTransformModel * glm::vec4(0, 0, 0.05f * delta.x, 1));
        }
    }

    drawTranslation(cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(), selectedTransform->getModelMatrix(),
                    cameraSystem->getNativeGraphicsMainFBO(), highlightedTransformAxis, selectedTransformAxis);
}

void TransformGizmo::updateRotation(PhysicsEngine::EditorCameraSystem *cameraSystem,
                                    PhysicsEngine::Transform *selectedTransform, float contentWidth,
                                    float contentHeight)
{
    drawRotation(cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(), selectedTransform->getModelMatrix(),
                 cameraSystem->getNativeGraphicsMainFBO(), highlightedTransformAxis, selectedTransformAxis);
}

void TransformGizmo::updateScale(PhysicsEngine::EditorCameraSystem *cameraSystem,
                                 PhysicsEngine::Transform *selectedTransform, float contentWidth, float contentHeight)
{
    drawScale(cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(), selectedTransform->getModelMatrix(),
              cameraSystem->getNativeGraphicsMainFBO(), highlightedTransformAxis, selectedTransformAxis);
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