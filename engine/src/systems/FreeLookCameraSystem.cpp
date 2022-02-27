#include "../../include/systems/FreeLookCameraSystem.h"

#include "../../include/core/World.h"

using namespace PhysicsEngine;

const float FreeLookCameraSystem::YAW_PAN_SENSITIVITY = 0.0025f;
const float FreeLookCameraSystem::PITCH_PAN_SENSITIVITY = 0.0025f;
const float FreeLookCameraSystem::ZOOM_SENSITIVITY = 0.2f;      // 125.0f;
const float FreeLookCameraSystem::TRANSLATE_SENSITIVITY = 1.0f; // 75.0f;

FreeLookCameraSystem::FreeLookCameraSystem(World* world) : System(world)
{
    mTransform = nullptr;
    mCamera = nullptr;

    mMousePosX = 0;
    mMousePosY = 0;
    mIsLeftMouseClicked = false;
    mIsRightMouseClicked = false;
    rotationOnClick = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
}

FreeLookCameraSystem::FreeLookCameraSystem(World* world, Guid id) : System(world, id)
{
    mMousePosX = 0;
    mMousePosY = 0;
    mIsLeftMouseClicked = false;
    mIsRightMouseClicked = false;
}

FreeLookCameraSystem::~FreeLookCameraSystem()
{
}

void FreeLookCameraSystem::serialize(YAML::Node& out) const
{
    System::serialize(out);
}

void FreeLookCameraSystem::deserialize(const YAML::Node& in)
{
    System::deserialize(in);
}

int FreeLookCameraSystem::getType() const
{
    return PhysicsEngine::FREELOOKCAMERASYSTEM_TYPE;
}

std::string FreeLookCameraSystem::getObjectName() const
{
    return PhysicsEngine::FREELOOKCAMERASYSTEM_NAME;
}

void FreeLookCameraSystem::init(World* world)
{
    mWorld = world;

    Entity* entity = world->createEntity();
    entity->mDoNotDestroy = true;
    entity->mHide = HideFlag::DontSave;

    mCamera = entity->addComponent<Camera>();
    mCamera->mHide = HideFlag::DontSave;

    mTransform = entity->getComponent<Transform>();
    mTransform->mPosition = glm::vec3(0, 2, -10);
    mHide = HideFlag::DontSave;
}

void FreeLookCameraSystem::update(const Input& input, const Time& time)
{
    glm::vec3 position = mTransform->mPosition;
    glm::vec3 front = mTransform->getForward();
    glm::vec3 up = mTransform->getUp();
    glm::vec3 right = mTransform->getRight();

    // D pad controls
    if (!getMouseButton(input, MouseButton::RButton))
    {
        if (getKey(input, KeyCode::Up))
        {
            position += FreeLookCameraSystem::ZOOM_SENSITIVITY * front;
        }
        if (getKey(input, KeyCode::Down))
        {
            position -= FreeLookCameraSystem::ZOOM_SENSITIVITY * front;
        }
        if (getKey(input, KeyCode::Left))
        {
            position += FreeLookCameraSystem::TRANSLATE_SENSITIVITY * right;
        }
        if (getKey(input, KeyCode::Right))
        {
            position -= FreeLookCameraSystem::TRANSLATE_SENSITIVITY * right;
        }
    }

    // WASD controls
    if (getMouseButton(input, MouseButton::RButton))
    {
        if (getKey(input, KeyCode::W))
        {
            position += FreeLookCameraSystem::ZOOM_SENSITIVITY * front;
        }
        if (getKey(input, KeyCode::S))
        {
            position -= FreeLookCameraSystem::ZOOM_SENSITIVITY * front;
        }
        if (getKey(input, KeyCode::A))
        {
            position += FreeLookCameraSystem::TRANSLATE_SENSITIVITY * right;
        }
        if (getKey(input, KeyCode::D))
        {
            position -= FreeLookCameraSystem::TRANSLATE_SENSITIVITY * right;
        }
    }

    // Mouse scroll wheel
    position += FreeLookCameraSystem::ZOOM_SENSITIVITY * input.mMouseDelta * front;

    // Mouse position
    mMousePosX = input.mMousePosX;
    mMousePosY = input.mMousePosY;

    // Mouse buttons
    mIsLeftMouseClicked = getMouseButtonDown(input, MouseButton::LButton);
    mIsRightMouseClicked = getMouseButtonDown(input, MouseButton::RButton);
    mIsLeftMouseHeldDown = getMouseButton(input, MouseButton::LButton);
    mIsRightMouseHeldDown = getMouseButton(input, MouseButton::RButton);

    if (mIsLeftMouseClicked)
    {
        mMousePosXOnLeftClick = mMousePosX;
        mMousePosYOnLeftClick = mMousePosY;
    }

    if (mIsRightMouseClicked)
    {
        mMousePosXOnRightClick = mMousePosX;
        mMousePosYOnRightClick = mMousePosY;
        rotationOnClick = mTransform->mRotation;
    }
    else if (mIsRightMouseHeldDown)
    {
        float yaw = FreeLookCameraSystem::YAW_PAN_SENSITIVITY * (mMousePosXOnRightClick - mMousePosX);
        float pitch = FreeLookCameraSystem::PITCH_PAN_SENSITIVITY * (mMousePosYOnRightClick - mMousePosY);

        // https://gamedev.stackexchange.com/questions/136174/im-rotating-an-object-on-two-axes-so-why-does-it-keep-twisting-around-the-thir
        mTransform->mRotation =
            glm::angleAxis(yaw, glm::vec3(0, 1, 0)) * rotationOnClick * glm::angleAxis(pitch, glm::vec3(1, 0, 0));
    }

    mCamera->computeViewMatrix(position, front, up, right);

    mTransform->mPosition = position;
}

void FreeLookCameraSystem::resetCamera()
{
    mTransform->mPosition = glm::vec3(0, 2, -10);
    mCamera->mBackgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
}

void FreeLookCameraSystem::configureCamera()
{
    mCamera->mRenderToScreen = true;
}

void FreeLookCameraSystem::setViewport(Viewport viewport)
{
    mCamera->setViewport(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);
}

void FreeLookCameraSystem::setFrustum(Frustum frustum)
{
    mCamera->setFrustum(frustum.mFov, frustum.mAspectRatio, frustum.mNearPlane, frustum.mFarPlane);
}

void FreeLookCameraSystem::setRenderPath(RenderPath path)
{
    mCamera->mRenderPath = path;
}

void FreeLookCameraSystem::setSSAO(CameraSSAO ssao)
{
    mCamera->mSSAO = ssao;
}

void FreeLookCameraSystem::setGizmos(CameraGizmos gizmos)
{
    mCamera->mGizmos = gizmos;
}

Viewport FreeLookCameraSystem::getViewport() const
{
    return mCamera->getViewport();
}

Frustum FreeLookCameraSystem::getFrustum() const
{
    return mCamera->getFrustum();
}

RenderPath FreeLookCameraSystem::getRenderPath() const
{
    return mCamera->mRenderPath;
}

CameraSSAO FreeLookCameraSystem::getSSAO() const
{
    return mCamera->mSSAO;
}

CameraGizmos FreeLookCameraSystem::getGizmos() const
{
    return mCamera->mGizmos;
}

Camera* FreeLookCameraSystem::getCamera() const
{
    return mCamera;
}

int FreeLookCameraSystem::getMousePosX() const
{
    return mMousePosX;
}

int FreeLookCameraSystem::getMousePosY() const
{
    return mMousePosY;
}

bool FreeLookCameraSystem::isLeftMouseClicked() const
{
    return mIsLeftMouseClicked;
}

bool FreeLookCameraSystem::isRightMouseClicked() const
{
    return mIsRightMouseClicked;
}

bool FreeLookCameraSystem::isLeftMouseHeldDown() const
{
    return mIsLeftMouseHeldDown;
}

bool FreeLookCameraSystem::isRightMouseHeldDown() const
{
    return mIsRightMouseHeldDown;
}

glm::vec2 FreeLookCameraSystem::distanceTraveledSinceLeftMouseClick() const
{
    return glm::vec2(mMousePosX - mMousePosXOnLeftClick, mMousePosY - mMousePosYOnLeftClick);
}

glm::vec2 FreeLookCameraSystem::distanceTraveledSinceRightMouseClick() const
{
    return glm::vec2(mMousePosX - mMousePosXOnRightClick, mMousePosY - mMousePosYOnRightClick);
}

Guid FreeLookCameraSystem::getTransformUnderMouse(float nx, float ny) const
{
    int x = (int)(mCamera->getViewport().mX + mCamera->getViewport().mWidth * nx);
    int y = (int)(mCamera->getViewport().mY + mCamera->getViewport().mHeight * ny);

    return mCamera->getTransformIdAtScreenPos(x, y);
}

unsigned int FreeLookCameraSystem::getNativeGraphicsMainFBO() const
{
    return mCamera->getNativeGraphicsMainFBO();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsColorTex() const
{
    return mCamera->getNativeGraphicsColorTex();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsDepthTex() const
{
    return mCamera->getNativeGraphicsDepthTex();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsColorPickingTex() const
{
    return mCamera->getNativeGraphicsColorPickingTex();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsPositionTex() const
{
    return mCamera->getNativeGraphicsPositionTex();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsNormalTex() const
{
    return mCamera->getNativeGraphicsNormalTex();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsAlbedoSpecTex() const
{
    return mCamera->getNativeGraphicsAlbedoSpecTex();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsSSAOColorTex() const
{
    return mCamera->getNativeGraphicsSSAOColorTex();
}

unsigned int FreeLookCameraSystem::getNativeGraphicsSSAONoiseTex() const
{
    return mCamera->getNativeGraphicsSSAONoiseTex();
}

GraphicsQuery FreeLookCameraSystem::getQuery() const
{
    return mCamera->mQuery;
}

glm::vec3 FreeLookCameraSystem::getCameraForward() const
{
    return mTransform->getForward();
}

glm::vec3 FreeLookCameraSystem::getCameraPosition() const
{
    return mCamera->getPosition();
}

glm::mat4 FreeLookCameraSystem::getViewMatrix() const
{
    return mCamera->getViewMatrix();
}

glm::mat4 FreeLookCameraSystem::getInvViewMatrix() const
{
    return mCamera->getInvViewMatrix();
}

glm::mat4 FreeLookCameraSystem::getProjMatrix() const
{
    return mCamera->getProjMatrix();
}

Ray FreeLookCameraSystem::normalizedDeviceSpaceToRay(float x, float y) const
{
    return mCamera->normalizedDeviceSpaceToRay(x, y);
}