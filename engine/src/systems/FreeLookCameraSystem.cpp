#include "../../include/systems/FreeLookCameraSystem.h"

#include "../../include/core/SerializationYaml.h"
#include "../../include/core/World.h"

#include <iostream>

using namespace PhysicsEngine;

const float FreeLookCameraSystem::YAW_PAN_SENSITIVITY = 0.0025f;
const float FreeLookCameraSystem::PITCH_PAN_SENSITIVITY = 0.0025f;
const float FreeLookCameraSystem::ZOOM_SENSITIVITY = 0.2f;      // 125.0f;
const float FreeLookCameraSystem::TRANSLATE_SENSITIVITY = 1.0f; // 75.0f;

FreeLookCameraSystem::FreeLookCameraSystem(World *world, const Id &id) : mWorld(world), mGuid(Guid::INVALID), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
    mTransformId = Guid::INVALID;
    mCameraId = Guid::INVALID;

    mMousePosX = 0;
    mMousePosY = 0;
    mIsLeftMouseClicked = false;
    mIsRightMouseClicked = false;
    rotationOnClick = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
}

FreeLookCameraSystem::FreeLookCameraSystem(World *world, const Guid &guid, const Id &id) : mWorld(world), mGuid(guid), mId(id), mHide(HideFlag::None)
{
    mEnabled = true;
    mTransformId = Guid::INVALID;
    mCameraId = Guid::INVALID;

    mMousePosX = 0;
    mMousePosY = 0;
    mIsLeftMouseClicked = false;
    mIsRightMouseClicked = false;
}

FreeLookCameraSystem::~FreeLookCameraSystem()
{
}

void FreeLookCameraSystem::serialize(YAML::Node &out) const
{
    out["type"] = getType();
    out["hide"] = mHide;
    out["id"] = mGuid;
}

void FreeLookCameraSystem::deserialize(const YAML::Node &in)
{
    mHide = YAML::getValue<HideFlag>(in, "hide");
    mGuid = YAML::getValue<Guid>(in, "id");
}

int FreeLookCameraSystem::getType() const
{
    return PhysicsEngine::FREELOOKCAMERASYSTEM_TYPE;
}

std::string FreeLookCameraSystem::getObjectName() const
{
    return PhysicsEngine::FREELOOKCAMERASYSTEM_NAME;
}

Guid FreeLookCameraSystem::getGuid() const
{
    return mGuid;
}

Id FreeLookCameraSystem::getId() const
{
    return mId;
}

void FreeLookCameraSystem::init(World *world)
{
    mWorld = world;

    Camera *camera = nullptr;
    Transform *transform = nullptr;

    if (mSpawnCameraOnInit)
    {
        Entity *entity = world->getActiveScene()->createEntity();
        entity->mDoNotDestroy = true;
        entity->mHide = HideFlag::DontSave;

        std::cout << "do not destroy camera entity id: " << entity->getGuid().toString() << std::endl;

        camera = entity->addComponent<Camera>();
        camera->mHide = HideFlag::DontSave;

        transform = entity->getComponent<Transform>();
        transform->setPosition(glm::vec3(0, 2, -10));
        transform->mHide = HideFlag::DontSave;
    }
    else
    {
        camera = mWorld->getActiveScene()->getComponentByIndex<Camera>(0);
        transform = camera->getComponent<Transform>();
    }

    std::string test = transform->getId().toString();

    //std::cout << "rotationOnInit: " << transform->getRotation().x << " " << transform->getRotation().y << " "
    //          << transform->getRotation().z << " " << transform->getRotation().w << " transform id: " << transform->getId().toString() << std::endl;

    camera->mRenderToScreen = mRenderToScreen;

    mCameraId = camera->getGuid();
    mTransformId = transform->getGuid();

    /*std::cout << "mWorld->getActiveScene()->getNumberOfComponents<Transform>(): "
              << mWorld->getActiveScene()->getNumberOfComponents<Transform>() 
              << " mWorld->getActiveScene()->getTransformDataCount(): "
              << mWorld->getActiveScene()->getTransformDataCount() << std::endl;

    std::cout << "transform guid: " << transform->getGuid().toString()
              << " transform id: " << transform->getId().toString() << std::endl;

    for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Transform>(); i++)
    {
        Transform *t = mWorld->getActiveScene()->getComponentByIndex<Transform>(i);
        std::cout << "t guid: " << t->getGuid().toString()
                  << " t id: " << t->getId().toString() << std::endl;
    }*/
}

void FreeLookCameraSystem::update(const Input &input, const Time &time)
{
    Camera *camera = getCamera();
    Transform *transform = getTransform();

    glm::vec3 position = transform->getPosition();
    glm::vec3 front = transform->getForward();
    glm::vec3 up = transform->getUp();
    glm::vec3 right = transform->getRight();

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
        rotationOnClick = transform->getRotation();
        /*std::cout << "rotationOnClick: " << rotationOnClick.x << " " << rotationOnClick.y << " " << rotationOnClick.z
                  << " " << rotationOnClick.w << " transform id: " << transform->getId().toString() << std::endl;
        
        std::cout << "mWorld->getActiveScene()->getNumberOfComponents<Transform>(): "
                  << mWorld->getActiveScene()->getNumberOfComponents<Transform>()
                  << " mWorld->getActiveScene()->getTransformDataCount(): "
                  << mWorld->getActiveScene()->getTransformDataCount() << std::endl;

        std::cout << "transform guid: " << transform->getGuid().toString()
                  << " transform id: " << transform->getId().toString() << std::endl;*/

        /*for (size_t i = 0; i < mWorld->getActiveScene()->getNumberOfComponents<Transform>(); i++)
        {
            Transform *t = mWorld->getActiveScene()->getComponentByIndex<Transform>(i);
            std::cout << "t guid: " << t->getGuid().toString() << " t id: " << t->getId().toString() << std::endl;
        }*/
    }
    else if (mIsRightMouseHeldDown)
    {
        float yaw = FreeLookCameraSystem::YAW_PAN_SENSITIVITY * (mMousePosXOnRightClick - mMousePosX);
        float pitch = FreeLookCameraSystem::PITCH_PAN_SENSITIVITY * (mMousePosYOnRightClick - mMousePosY);

        // https://gamedev.stackexchange.com/questions/136174/im-rotating-an-object-on-two-axes-so-why-does-it-keep-twisting-around-the-thir
        // mTransform->mRotation =
        //    glm::angleAxis(yaw, glm::vec3(0, 1, 0)) * rotationOnClick * glm::angleAxis(pitch, glm::vec3(1, 0, 0));
        transform->setRotation(glm::angleAxis(yaw, glm::vec3(0, 1, 0)) * rotationOnClick * glm::angleAxis(pitch, glm::vec3(1, 0, 0)));
    
        //glm::quat temp =
        //    glm::angleAxis(yaw, glm::vec3(0, 1, 0)) * rotationOnClick * glm::angleAxis(pitch, glm::vec3(1, 0, 0));


        //std::cout << "Editor camera rotation: " << transform->getRotation().x << " " << transform->getRotation().y
        //          << " " << transform->getRotation().z << " " << transform->getRotation().w << " " << temp.x << " "
        //          << temp.y << " " << temp.z << " " << temp.w << " yaw: " << yaw
        //          << " pitch: " << pitch << std::endl;
    }

    camera->computeViewMatrix(position, front, up, right);

    transform->setPosition(position);
}

void FreeLookCameraSystem::resetCamera()
{
    getTransform()->setPosition(glm::vec3(0, 2, -10));
    getCamera()->mBackgroundColor = glm::vec4(0.15, 0.15f, 0.15f, 1.0f);
}

void FreeLookCameraSystem::configureCamera(CameraSystemConfig config)
{
    mRenderToScreen = config.mRenderToScreen;
    mSpawnCameraOnInit = config.mSpawnCameraOnInit;
}

void FreeLookCameraSystem::setViewport(const Viewport &viewport)
{
    getCamera()->setViewport(viewport.mX, viewport.mY, viewport.mWidth, viewport.mHeight);
}

void FreeLookCameraSystem::setFrustum(const Frustum &frustum)
{
    getCamera()->setFrustum(frustum.mFov, frustum.mAspectRatio, frustum.mNearPlane, frustum.mFarPlane);
}

void FreeLookCameraSystem::setViewport(int x, int y, int width, int height)
{
    getCamera()->setViewport(x, y, width, height);
}

void FreeLookCameraSystem::setFrustum(float fov, float aspectRatio, float near, float far)
{
    getCamera()->setFrustum(fov, aspectRatio, near, far);
}

void FreeLookCameraSystem::setRenderPath(RenderPath path)
{
    getCamera()->mRenderPath = path;
}

void FreeLookCameraSystem::setSSAO(CameraSSAO ssao)
{
    getCamera()->mSSAO = ssao;
}

void FreeLookCameraSystem::setGizmos(CameraGizmos gizmos)
{
    getCamera()->mGizmos = gizmos;
}

Viewport FreeLookCameraSystem::getViewport() const
{
    return getCamera()->getViewport();
}

Frustum FreeLookCameraSystem::getFrustum() const
{
    return getCamera()->getFrustum();
}

RenderPath FreeLookCameraSystem::getRenderPath() const
{
    return getCamera()->mRenderPath;
}

CameraSSAO FreeLookCameraSystem::getSSAO() const
{
    return getCamera()->mSSAO;
}

CameraGizmos FreeLookCameraSystem::getGizmos() const
{
    return getCamera()->mGizmos;
}

Camera *FreeLookCameraSystem::getCamera() const
{
    return mWorld->getActiveScene()->getComponentByGuid<Camera>(mCameraId);
}

Transform *FreeLookCameraSystem::getTransform() const
{
    return mWorld->getActiveScene()->getComponentByGuid<Transform>(mTransformId);
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

Id FreeLookCameraSystem::getTransformUnderMouse(float nx, float ny) const
{
    Camera *camera = getCamera();
    int x = (int)(camera->getViewport().mX + camera->getViewport().mWidth * nx);
    int y = (int)(camera->getViewport().mY + camera->getViewport().mHeight * ny);

    return camera->getTransformIdAtScreenPos(x, y);
}

Framebuffer *FreeLookCameraSystem::getNativeGraphicsMainFBO() const
{
    return getCamera()->getNativeGraphicsMainFBO();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsColorTex() const
{
    return getCamera()->getNativeGraphicsColorTex();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsDepthTex() const
{
    return getCamera()->getNativeGraphicsDepthTex();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsColorPickingTex() const
{
    return getCamera()->getNativeGraphicsColorPickingTex();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsPositionTex() const
{
    return getCamera()->getNativeGraphicsPositionTex();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsNormalTex() const
{
    return getCamera()->getNativeGraphicsNormalTex();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsAlbedoSpecTex() const
{
    return getCamera()->getNativeGraphicsAlbedoSpecTex();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsSSAOColorTex() const
{
    return getCamera()->getNativeGraphicsSSAOColorTex();
}

RenderTextureHandle *FreeLookCameraSystem::getNativeGraphicsSSAONoiseTex() const
{
    return getCamera()->getNativeGraphicsSSAONoiseTex();
}

GraphicsQuery FreeLookCameraSystem::getQuery() const
{
    return getCamera()->mQuery;
}

glm::vec3 FreeLookCameraSystem::getCameraForward() const
{
    return getTransform()->getForward();
}

glm::vec3 FreeLookCameraSystem::getCameraPosition() const
{
    return getCamera()->getPosition();
}

glm::mat4 FreeLookCameraSystem::getViewMatrix() const
{
    return getCamera()->getViewMatrix();
}

glm::mat4 FreeLookCameraSystem::getInvViewMatrix() const
{
    return getCamera()->getInvViewMatrix();
}

glm::mat4 FreeLookCameraSystem::getProjMatrix() const
{
    return getCamera()->getProjMatrix();
}

Ray FreeLookCameraSystem::normalizedDeviceSpaceToRay(float x, float y) const
{
    return getCamera()->normalizedDeviceSpaceToRay(x, y);
}