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
			std::vector<bool> keyIsDown;
			std::vector<bool> keyWasDown;
			std::vector<bool> buttonIsDown;
			std::vector<bool> buttonWasDown;

			int mousePosX;
			int mousePosY;
			int mouseDelta;
			
		public:
			Input();
			~Input();

		public:
			bool getKey(KeyCode key);
			bool getKeyDown(KeyCode key);
			bool getKeyUp(KeyCode key);

			bool getMouseButton(MouseButton button);
			bool getMouseButtonDown(MouseButton button);
			bool getMouseButtonUp(MouseButton button);
			int getMousePosX();
			int getMousePosY();
			int getMouseDelta();

			void setKeyState(KeyCode key, bool isDown, bool wasDown);
			void setMouseButtonState(MouseButton button, bool isDown, bool wasDown);
			void setMousePosition(int x, int y);
			void setMouseDelta(int delta);
	};
}


#endif