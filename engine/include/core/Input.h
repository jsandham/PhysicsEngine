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
    Clear = 50,
    Menu = 51,
    Pause = 52,
    PrintScreen = 53,
    Insert = 54,
    Delete = 55,
    Help = 56,
    NumPad0 = 57,
    NumPad1 = 58,
    NumPad2 = 59,
    NumPad3 = 60,
    NumPad4 = 61,
    NumPad5 = 62,
    NumPad6 = 63,
    NumPad7 = 64,
    NumPad8 = 65,
    NumPad9 = 66,
    NumPadMultiply = 67,
    NumPadAdd = 68,
    NumPadSubtract = 69,
    NumPadDivide = 70,
    F1 = 71,
    F2 = 72,
    F3 = 73,
    F4 = 74,
    F5 = 75,
    F6 = 76,
    F7 = 77,
    F8 = 78,
    F9 = 79,
    F10 = 80,
    F11 = 81,
    F12 = 82,
    Invalid = 83,
    Count = 84
};

enum class MouseButton
{
    LButton = 0,
    MButton = 1,
    RButton = 2,
    Alt0Button = 3,
    Alt1Button = 4,
    Count = 5
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
    YButton = 13,
    Count = 14
};

struct Input
{
    bool mKeyIsDown[84];
    bool mKeyWasDown[84];
    bool mMouseButtonIsDown[5];
    bool mMouseButtonWasDown[5];
    bool mXboxButtonIsDown[14];
    bool mXboxButtonWasDown[14];
    int mMousePosX;
    int mMousePosY;
    float mMouseDelta;
    float mMouseDeltaH; // horizontal scroll
    int mLeftStickX;
    int mLeftStickY;
    int mRightStickX;
    int mRightStickY;
};

Input& getInput();

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