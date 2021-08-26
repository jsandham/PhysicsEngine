#ifndef INPUT_H__
#define INPUT_H__

namespace PhysicsEngine
{
enum class KeyCode
{
    Key0 = 0,
    Key1 = 1,
    Key2 = 2,
    Key3 = 3,
    Key4 = 4,
    Key5 = 5,
    Key6 = 6,
    Key7 = 7,
    Key8 = 8,
    Key9 = 9,
    A = 10,
    B = 11,
    C = 12,
    D = 13,
    E = 14,
    F = 15,
    G = 16,
    H = 17,
    I = 18,
    J = 19,
    K = 20,
    L = 21,
    M = 22,
    N = 23,
    O = 24,
    P = 25,
    Q = 26,
    R = 27,
    S = 28,
    T = 29,
    U = 30,
    V = 31,
    W = 32,
    X = 33,
    Y = 34,
    Z = 35,
    Enter = 36,
    Up = 37,
    Down = 38,
    Left = 39,
    Right = 40,
    Space = 41,
    LShift = 42,
    RShift = 43,
    Tab = 44,
    Backspace = 45,
    CapsLock = 46,
    LCtrl = 47,
    RCtrl = 48,
    Escape = 49,
    NumPad0 = 50,
    NumPad1 = 51,
    NumPad2 = 52,
    NumPad3 = 53,
    NumPad4 = 54,
    NumPad5 = 55,
    NumPad6 = 56,
    NumPad7 = 57,
    NumPad8 = 58,
    NumPad9 = 59,
    Invalid = 60
};

enum class MouseButton
{
    LButton = 0,
    MButton = 1,
    RButton = 2,
    Alt0Button = 3,
    Alt1Button = 4
};

enum class XboxButton
{
    LeftDPad = 0,
    RightDPad = 1,
    UpDPad = 2,
    DownDPad = 3,
    Start = 4,
    Back = 5,
    LeftThumb = 6,
    RightThumb = 7,
    LeftShoulder = 8,
    RightShoulder = 9,
    AButton = 10,
    BButton = 11,
    XButton = 12,
    YButton = 13
};

struct Input
{
    bool mKeyIsDown[61];
    bool mKeyWasDown[61];
    bool mMouseButtonIsDown[5];
    bool mMouseButtonWasDown[5];
    bool mXboxButtonIsDown[14];
    bool mXboxButtonWasDown[14];
    int mMousePosX;
    int mMousePosY;
    int mMouseDelta;
    int mLeftStickX;
    int mLeftStickY;
    int mRightStickX;
    int mRightStickY;
};

bool getKey(const Input &input, KeyCode key);
bool getKeyDown(const Input &input, KeyCode key);
bool getKeyUp(const Input &input, KeyCode key);
bool getMouseButton(const Input &input, MouseButton button);
bool getMouseButtonDown(const Input &input, MouseButton button);
bool getMouseButtonUp(const Input &input, MouseButton button);
bool getXboxButton(const Input &input, XboxButton button);
bool getXboxButtonDown(const Input &input, XboxButton button);
bool getXboxButtonUp(const Input &input, XboxButton button);
} // namespace PhysicsEngine

#endif