#include <core/Application.h>
#define PHYSICSENGINE_PLATFORM_WINDOWS 1
#include <core/EntryPoint.h>
#include "include/EditorLayer.h"
#include "include/ImGuiLayer.h"

namespace PhysicsEditor
{
	class EditorApplication : public PhysicsEngine::Application
	{
	private:
		ImGuiLayer mImguiLayer;
		EditorLayer mEditorLayer;

	public:
		EditorApplication() : Application("PhysicsEditor", 1536, 864)
		{
			pushLayer(&mImguiLayer);
			pushLayer(&mEditorLayer);
		}

		~EditorApplication()
		{
		}
	};
}

PhysicsEngine::Application* PhysicsEngine::createApplication()
{
	return new PhysicsEditor::EditorApplication();
}

#ifdef PHYSICSENGINE_PLATFORM_WINDOWS

#include <Windows.h>
#include <ShObjIdl.h>

extern LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
LRESULT Application_WndProcHandler(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
	return ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam);
}

std::string openFile(const char* filter)
{
	OPENFILENAMEA ofn;
	char szFile[260] = {0};
	ZeroMemory(&ofn, sizeof(OPENFILENAMEA));
	ofn.lStructSize = sizeof(OPENFILENAMEA);
	ofn.hwndOwner = static_cast<HWND>(PhysicsEngine::Application::get().getWindow().getNativeWindow());
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = filter;
	ofn.nFilterIndex = 1;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
	if(GetOpenFileName(&ofn) == TRUE)
	{
		return ofn.lpstrFile;
	}
	return std::string();
}

std::string saveFile(const char* filter)
{
	OPENFILENAMEA ofn;
	CHAR szFile[260] = { 0 };
	ZeroMemory(&ofn, sizeof(OPENFILENAMEA));
	ofn.lStructSize = sizeof(OPENFILENAMEA);
	ofn.hwndOwner = static_cast<HWND>(PhysicsEngine::Application::get().getWindow().getNativeWindow());
	ofn.lpstrFile = szFile;
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = filter;
	ofn.nFilterIndex = 1;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;
	if (GetSaveFileName(&ofn) == TRUE)
	{
		return ofn.lpstrFile;
	}
	return std::string();
}

std::string selectFolder()
{
	std::string folderPath = std::string();

	HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED |
		COINIT_DISABLE_OLE1DDE);
	if (SUCCEEDED(hr))
	{
		IFileOpenDialog* pFileOpen;

		// Create the FileOpenDialog object.
		hr = CoCreateInstance(CLSID_FileOpenDialog, NULL, CLSCTX_ALL,
			IID_IFileOpenDialog, reinterpret_cast<void**>(&pFileOpen));

		if (SUCCEEDED(hr))
		{
			DWORD dwOptions;
			if (SUCCEEDED(pFileOpen->GetOptions(&dwOptions)))
			{
				pFileOpen->SetOptions(dwOptions | FOS_PICKFOLDERS);
			}

			// Show the Open dialog box.
			hr = pFileOpen->Show(NULL);

			// Get the file name from the dialog box.
			if (SUCCEEDED(hr))
			{
				IShellItem* pItem;
				hr = pFileOpen->GetResult(&pItem);
				if (SUCCEEDED(hr))
				{
					PWSTR pszFilePath;
					hr = pItem->GetDisplayName(SIGDN_FILESYSPATH, &pszFilePath);

					// Display the file name to the user.
					if (SUCCEEDED(hr))
					{
						char str[256];
						wcstombs(str, pszFilePath, 256);
						folderPath = std::string(str);
						CoTaskMemFree(pszFilePath);
					}
					pItem->Release();
				}
			}
			pFileOpen->Release();
		}
		CoUninitialize();
	}

	return folderPath;
}
#endif
