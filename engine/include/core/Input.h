#ifndef __INPUT_H__
#define __INPUT_H__

#include <vector>

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

	class Input
	{
		private:
			static std::vector<bool> keyIsDown;
			static std::vector<bool> keyWasDown;
			static std::vector<bool> buttonIsDown;
			static std::vector<bool> buttonWasDown;
			static int mousePosX;
			static int mousePosY;
			static int mouseDelta;
			
		private:
			Input();

		public:
			static bool getKey(KeyCode key);
			static bool getKeyDown(KeyCode key);
			static bool getKeyUp(KeyCode key);

			static bool getMouseButton(MouseButton button);
			static bool getMouseButtonDown(MouseButton button);
			static bool getMouseButtonUp(MouseButton button);
			static int getMousePosX();
			static int getMousePosY();
			static int getMouseDelta();

			static void setKeyState(KeyCode key, bool isDown, bool wasDown);
			static void setMouseButtonState(MouseButton button, bool isDown, bool wasDown);
			static void setMousePosition(int x, int y);
			static void setMouseDelta(int delta);
			static void updateEOF();
	};
}


#endif