#include "core/ClosestDistance.h"
#include "core/InternalMeshes.h"
#include "core/Intersect.h"
#include "graphics/Graphics.h"

#include "../include/TransformGizmo.h"

#include "imgui.h"

using namespace PhysicsEditor;
using namespace PhysicsEngine;

void TransformGizmo::initialize()
{
    const std::string gizmoVertShader = "#version 430 core\n"
                                        "uniform mat4 mvp;\n"
                                        "in vec3 position;\n"
                                        "void main()\n"
                                        "{\n"
                                        "	gl_Position = mvp * vec4(position, 1.0);\n"
                                        "}";

    const std::string gizmoFragShader = "#version 430 core\n"
                                        "uniform vec4 color;\n"
                                        "out vec4 FragColor;\n"
                                        "void main()\n"
                                        "{\n"
                                        "	FragColor = color;\n"
                                        "}";

    const std::string gizmo3dVertShader = "#version 430 core\n"
                                          "layout(location = 0) in vec3 position;\n"
                                          "layout(location = 1) in vec3 normal;\n"
                                          "out vec3 FragPos;\n"
                                          "out vec3 Normal;\n"
                                          "uniform mat4 model;\n"
                                          "uniform mat4 view;\n"
                                          "uniform mat4 projection;\n"
                                          "void main()\n"
                                          "{\n"
                                          "    FragPos = vec3(model * vec4(position, 1.0));\n"
                                          "    Normal = mat3(transpose(inverse(model))) * normal;\n"
                                          "    gl_Position = projection * view * vec4(FragPos, 1.0);\n"
                                          "}";

    const std::string gizmo3dFragShader = "#version 430 core\n"
                                          "out vec4 FragColor;\n"
                                          "in vec3 Normal;\n"
                                          "in vec3 FragPos;\n"
                                          "uniform vec3 lightPos;\n"
                                          "uniform vec4 color;\n"
                                          "void main()\n"
                                          "{\n"
                                          "    vec3 norm = normalize(Normal);\n"
                                          "    vec3 lightDir = normalize(lightPos - FragPos);\n"
                                          "    float diff = max(abs(dot(norm, lightDir)), 0.1);\n"
                                          "    vec4 diffuse = vec4(diff, diff, diff, 1.0);\n"
                                          "    FragColor = diffuse * color;\n"
                                          "}";

    Graphics::compile(gizmoVertShader, gizmoFragShader, "", &mGizmoShaderProgram);
    Graphics::compile(gizmo3dVertShader, gizmo3dFragShader, "", &mGizmo3dShaderProgram);

    mGizmoShaderMVPLoc = Graphics::findUniformLocation("mvp", mGizmoShaderProgram);
    mGizmoShaderColorLoc = Graphics::findUniformLocation("color", mGizmoShaderProgram);

    mGizmo3dShaderModelLoc = Graphics::findUniformLocation("model", mGizmo3dShaderProgram);
    mGizmo3dShaderViewLoc = Graphics::findUniformLocation("view", mGizmo3dShaderProgram);
    mGizmo3dShaderProjLoc = Graphics::findUniformLocation("projection", mGizmo3dShaderProgram);
    mGizmo3dShaderColorLoc = Graphics::findUniformLocation("color", mGizmo3dShaderProgram);
    mGizmo3dShaderLightPosLoc = Graphics::findUniformLocation("lightPos", mGizmo3dShaderProgram);

    // Set up translation gizmo vertex buffers
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

        Graphics::checkError(__LINE__, __FILE__);

        highlightedTransformAxis = Axis::Axis_None;
        selectedTransformAxis = Axis::Axis_None;
    }

    // Set up rotation gizmo vertex buffers
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

    // Set up scale gizmo vertex buffers etc
    for (int i = 0; i < 3; i++)
    {
        std::vector<float> scaleVertices = InternalMeshes::cubeVertices;
        std::vector<float> scaleNormals = InternalMeshes::cubeNormals;

        for (size_t j = 0; j < scaleVertices.size() / 3; j++)
        {
            scaleVertices[3 * j + 0] *= 0.1f;
            scaleVertices[3 * j + 1] *= 0.1f;
            scaleVertices[3 * j + 2] *= 0.1f;

            scaleVertices[3 * j + i] += 1.0f;
        }

        glGenVertexArrays(1, &mScaleVAO[i]);
        glBindVertexArray(mScaleVAO[i]);

        glGenBuffers(2, &mScaleVBO[2 * i]);
        glBindBuffer(GL_ARRAY_BUFFER, mScaleVBO[2 * i]);
        glBufferData(GL_ARRAY_BUFFER, scaleVertices.size() * sizeof(float), scaleVertices.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void *)0);

        glGenBuffers(1, &mScaleVBO[2 * i + 1]);
        glBindBuffer(GL_ARRAY_BUFFER, mScaleVBO[2 * i + 1]);
        glBufferData(GL_ARRAY_BUFFER, scaleNormals.size() * sizeof(float), scaleNormals.data(), GL_STATIC_DRAW);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GL_FLOAT), 0);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    glEnable(GL_LINE_SMOOTH);

    Graphics::checkError(__LINE__, __FILE__);

    mode = GizmoMode::Translation;
}

void TransformGizmo::update(PhysicsEngine::EditorCameraSystem *cameraSystem, PhysicsEngine::GizmoSystem *gizmoSystem,
                            PhysicsEngine::Transform *selectedTransform, float mousePosX, float mousePosY,
                            float contentWidth, float contentHeight)
{
    assert(cameraSystem != NULL);
    assert(selectedTransform != NULL);

    switch (mode)
    {
    case GizmoMode::Translation:
        updateTranslation(cameraSystem, selectedTransform, mousePosX, mousePosY, contentWidth, contentHeight);
        break;
    case GizmoMode::Rotation:
        updateRotation(cameraSystem, gizmoSystem, selectedTransform, mousePosX, mousePosY, contentWidth, contentHeight);
        break;
    case GizmoMode::Scale:
        updateScale(cameraSystem, selectedTransform, mousePosX, mousePosY, contentWidth, contentHeight);
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
                                       PhysicsEngine::Transform *selectedTransform, float mousePosX, float mousePosY,
                                       float contentWidth, float contentHeight)
{
    glm::mat4 model = selectedTransform->getModelMatrix();
    glm::vec3 position = glm::vec3(model[3]);

    glm::vec3 xnormal = glm::normalize(glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
    glm::vec3 ynormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
    glm::vec3 znormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));

    Ray xAxisRay(position, xnormal);
    Ray yAxisRay(position, ynormal);
    Ray zAxisRay(position, znormal);

    float ndcX = 2 * (mousePosX - 0.5f * contentWidth) / contentWidth;
    float ndcY = 2 * (mousePosY - 0.5f * contentHeight) / contentHeight;

    Ray cameraRay = cameraSystem->normalizedDeviceSpaceToRay(ndcX, ndcY);

    float t = 0.0f;
    float xt = 0.0f;
    float yt = 0.0f;
    float zt = 0.0f;

    float closestDistanceToAxisX = ClosestDistance::closestDistance(cameraRay, xAxisRay, t, xt);
    float closestDistanceToAxisY = ClosestDistance::closestDistance(cameraRay, yAxisRay, t, yt);
    float closestDistanceToAxisZ = ClosestDistance::closestDistance(cameraRay, zAxisRay, t, zt);

    highlightedTransformAxis = Axis::Axis_None;
    if (xt >= 0.0f && xt <= 1.0f && closestDistanceToAxisX < closestDistanceToAxisY &&
        closestDistanceToAxisX < closestDistanceToAxisZ)
    {
        highlightedTransformAxis = closestDistanceToAxisX < 0.05f ? Axis::Axis_X : highlightedTransformAxis;
    }
    else if (yt >= 0.0f && yt <= 1.0f && closestDistanceToAxisY < closestDistanceToAxisX &&
             closestDistanceToAxisY < closestDistanceToAxisZ)
    {
        highlightedTransformAxis = closestDistanceToAxisY < 0.05f ? Axis::Axis_Y : highlightedTransformAxis;
    }
    else if (zt >= 0.0f && zt <= 1.0f && closestDistanceToAxisZ < closestDistanceToAxisX &&
             closestDistanceToAxisZ < closestDistanceToAxisY)
    {
        highlightedTransformAxis = closestDistanceToAxisZ < 0.05f ? Axis::Axis_Z : highlightedTransformAxis;
    }

    ImGuiIO &io = ImGui::GetIO();

    if (io.MouseDown[0])
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
    else
    {
        selectedTransformAxis = Axis::Axis_None;
    }

    drawTranslation(cameraSystem->getViewport(), cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(),
                    selectedTransform->getModelMatrix(), cameraSystem->getNativeGraphicsMainFBO(),
                    highlightedTransformAxis, selectedTransformAxis);
}

void TransformGizmo::updateRotation(PhysicsEngine::EditorCameraSystem *cameraSystem,
                                    PhysicsEngine::GizmoSystem *gizmoSystem,
                                    PhysicsEngine::Transform *selectedTransform, float mousePosX, float mousePosY,
                                    float contentWidth, float contentHeight)
{
    glm::mat4 model = selectedTransform->getModelMatrix();
    glm::vec3 position = glm::vec3(model[3]);

    glm::vec3 xnormal = glm::normalize(glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 0.0f)));
    glm::vec3 ynormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
    glm::vec3 znormal = glm::normalize(glm::vec3(model * glm::vec4(0.0f, 0.0f, 1.0f, 0.0f)));

    Circle xAxisCircle(position, xnormal, 1.0f);
    Circle yAxisCircle(position, ynormal, 1.0f);
    Circle zAxisCircle(position, znormal, 1.0f);

    float ndcX = 2 * (mousePosX - 0.5f * contentWidth) / contentWidth;
    float ndcY = 2 * (mousePosY - 0.5f * contentHeight) / contentHeight;

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

    ImGuiIO &io = ImGui::GetIO();

    if (io.MouseDown[0])
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
    else
    {
        selectedTransformAxis = Axis::Axis_None;
    }

    drawRotation(cameraSystem->getViewport(), cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(),
                 selectedTransform->getModelMatrix(), cameraSystem->getNativeGraphicsMainFBO(),
                 highlightedTransformAxis, selectedTransformAxis);
}

void TransformGizmo::updateScale(PhysicsEngine::EditorCameraSystem *cameraSystem,
                                 PhysicsEngine::Transform *selectedTransform, float mousePosX, float mousePosY,
                                 float contentWidth, float contentHeight)
{
    float ndcX = 2 * (mousePosX - 0.5f * contentWidth) / contentWidth;
    float ndcY = 2 * (mousePosY - 0.5f * contentHeight) / contentHeight;

    Ray cameraRay = cameraSystem->normalizedDeviceSpaceToRay(ndcX, ndcY);

    glm::mat4 model = selectedTransform->getModelMatrix();

    glm::vec3 xcentre = glm::vec3(model * glm::vec4(1.0f, 0.0f, 0.0f, 1.0f));
    glm::vec3 ycentre = glm::vec3(model * glm::vec4(0.0f, 1.0f, 0.0f, 1.0f));
    glm::vec3 zcentre = glm::vec3(model * glm::vec4(0.0f, 0.0f, 1.0f, 1.0f));

    Sphere xAxisSphere(xcentre, 0.1f);
    Sphere yAxisSphere(ycentre, 0.1f);
    Sphere zAxisSphere(zcentre, 0.1f);

    float t = 0.0f;
    glm::vec3 xspherePoint;
    glm::vec3 yspherePoint;
    glm::vec3 zspherePoint;

    float closestDistanceToAxisX = ClosestDistance::closestDistance(cameraRay, xAxisSphere, t, xspherePoint);
    float closestDistanceToAxisY = ClosestDistance::closestDistance(cameraRay, yAxisSphere, t, yspherePoint);
    float closestDistanceToAxisZ = ClosestDistance::closestDistance(cameraRay, zAxisSphere, t, zspherePoint);

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

    ImGuiIO &io = ImGui::GetIO();

    if (io.MouseDown[0])
    {
        if (selectedTransformAxis == Axis::Axis_None)
        {
            selectedTransformAxis = highlightedTransformAxis;

            if (selectedTransformAxis == Axis::Axis_X)
            {
                start = xspherePoint;
            }
            else if (selectedTransformAxis == Axis::Axis_Y)
            {
                start = yspherePoint;
            }
            else if (selectedTransformAxis == Axis::Axis_Z)
            {
                start = zspherePoint;
            }
        }

        glm::vec3 delta = glm::vec3(0.0f, 0.0f, 0.0f);
        if (selectedTransformAxis == Axis::Axis_X)
        {
            end = xspherePoint;
            delta = end - start;
            start = end;
        }
        else if (selectedTransformAxis == Axis::Axis_Y)
        {
            end = yspherePoint;
            delta = end - start;
            start = end;
        }
        else if (selectedTransformAxis == Axis::Axis_Z)
        {
            end = zspherePoint;
            delta = end - start;
            start = end;
        }

        selectedTransform->mScale += delta;
    }
    else
    {
        selectedTransformAxis = Axis::Axis_None;
    }

    drawScale(cameraSystem->getViewport(), cameraSystem->getProjMatrix(), cameraSystem->getViewMatrix(),
              cameraSystem->getCameraPosition(), selectedTransform->getModelMatrix(),
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

void TransformGizmo::drawTranslation(const Viewport &viewport, const glm::mat4 &projection, const glm::mat4 &view,
                                     const glm::mat4 &model, GLuint fbo, Axis highlightAxis, Axis selectedAxis)
{
    glViewport(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);
    glScissor(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClear(GL_DEPTH_BUFFER_BIT);

    glm::mat4 mvp = projection * view * model;

    Graphics::use(mGizmoShaderProgram);
    Graphics::setMat4(mGizmoShaderMVPLoc, mvp);

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

        Graphics::setVec4(mGizmoShaderColorLoc, axis_color);

        glBindVertexArray(mTranslationVAO[i]);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void TransformGizmo::drawRotation(const Viewport &viewport, const glm::mat4 &projection, const glm::mat4 &view,
                                  const glm::mat4 &model, GLuint fbo, Axis highlightAxis, Axis selectedAxis)
{
    glViewport(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);
    glScissor(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClear(GL_DEPTH_BUFFER_BIT);

    glm::mat4 mvp = projection * view * model;

    Graphics::use(mGizmoShaderProgram);
    Graphics::setMat4(mGizmoShaderMVPLoc, mvp);

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

        Graphics::setVec4(mGizmoShaderColorLoc, axis_color);

        glBindVertexArray(mRotationVAO[i]);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 120);
        glBindVertexArray(0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void TransformGizmo::drawScale(const Viewport &viewport, const glm::mat4 &projection, const glm::mat4 &view,
                               const glm::vec3 &cameraPos, const glm::mat4 &model, GLuint fbo, Axis highlightAxis,
                               Axis selectedAxis)
{
    glViewport(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);
    glScissor(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);

    glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glClear(GL_DEPTH_BUFFER_BIT);

    glm::mat4 mvp = projection * view * model;

    Graphics::use(mGizmoShaderProgram);
    Graphics::setMat4(mGizmoShaderMVPLoc, mvp);

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

        Graphics::setVec4(mGizmoShaderColorLoc, axis_color);

        glBindVertexArray(mTranslationVAO[i]);
        glLineWidth(2.0f);
        glDrawArrays(GL_LINES, 0, 2);
        glBindVertexArray(0);
    }

    Graphics::use(mGizmo3dShaderProgram);
    Graphics::setMat4(mGizmo3dShaderModelLoc, model);
    Graphics::setMat4(mGizmo3dShaderViewLoc, view);
    Graphics::setMat4(mGizmo3dShaderProjLoc, projection);
    Graphics::setVec3(mGizmo3dShaderLightPosLoc, cameraPos);

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

        Graphics::setVec4(mGizmo3dShaderColorLoc, axis_color);

        glBindVertexArray(mScaleVAO[i]);
        glDrawArrays(GL_TRIANGLES, 0, 3 * 36);
        glBindVertexArray(0);
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}