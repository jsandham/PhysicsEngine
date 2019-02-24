#ifndef __INPUT_H__
#define __INPUT_H__

namespace PhysicsEngine
{
	typedef enum KeyCode
	{
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
	};

	typedef enum MouseButton
	{
		LButton,
		MButton,
		RButton
	};

	struct Input
	{
		bool keyIsDown[51];
		bool keyWasDown[51];
		bool buttonIsDown[3];
		bool buttonWasDown[3];
		int mousePosX;
		int mousePosY;
		int mouseDelta;
	};

	bool getKey(Input input, KeyCode key);
	bool getKeyDown(Input input, KeyCode key);
	bool getKeyUp(Input input, KeyCode key);
	bool getMouseButton(Input input, MouseButton button);
	bool getMouseButtonDown(Input input, MouseButton button);
	bool getMouseButtonUp(Input input, MouseButton button);
}


#endif