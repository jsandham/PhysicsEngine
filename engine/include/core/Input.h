#ifndef __INPUT_H__
#define __INPUT_H__

namespace PhysicsEngine
{
	typedef enum KeyCode
	{
		Key0,
		Key1,
		Key2,
		Key3,
		Key4,
		Key5,
		Key6,
		Key7,
		Key8,
		Key9,
		A,
		B,
		C,
		D,
		E,
		F,
		G,
		H,
		I,
		J,
		K,
		L,
		M,
		N,
		O,
		P,
		Q,
		R,
		S,
		T,
		U,
		V,
		W,
		X,
		Y,
		Z,
		Enter,
		Up,
		Down,
		Left,
		Right,
		Space,
		LShift,
		RShift,
		Tab,
		Backspace,
		CapsLock,
		LCtrl,
		RCtrl,
		Escape,
		NumPad0,
		NumPad1,
		NumPad2,
		NumPad3,
		NumPad4,
		NumPad5,
		NumPad6,
		NumPad7,
		NumPad8,
		NumPad9,
		Invalid
	}KeyCode;

	typedef enum MouseButton
	{
		LButton,
		MButton,
		RButton,
		Alt0Button,
		Alt1Button
	}MouseButton;

	typedef enum XboxButton
	{
		LeftDPad,
		RightDPad,
		UpDPad,
		DownDPad,
		Start,
		Back,
		LeftThumb,
		RightThumb,
		LeftShoulder,
		RightShoulder,
		AButton,
		BButton,
		XButton,
		YButton
	}XboxButton;

	struct Input
	{
		bool keyIsDown[61];
		bool keyWasDown[61];
		bool mouseButtonIsDown[5];
		bool mouseButtonWasDown[5];
		bool xboxButtonIsDown[14];
		bool xboxButtonWasDown[14];
		int mousePosX;
		int mousePosY;
		int mouseDelta;
		int leftStickX;
		int leftStickY;
		int rightStickX;
		int rightStickY;
	};

	bool getKey(Input input, KeyCode key);
	bool getKeyDown(Input input, KeyCode key);
	bool getKeyUp(Input input, KeyCode key);
	bool getMouseButton(Input input, MouseButton button);
	bool getMouseButtonDown(Input input, MouseButton button);
	bool getMouseButtonUp(Input input, MouseButton button);
	bool getXboxButton(Input input, XboxButton button);
	bool getXboxButtonDown(Input input, XboxButton button);
	bool getXboxButtonUp(Input input, XboxButton button);
}


#endif