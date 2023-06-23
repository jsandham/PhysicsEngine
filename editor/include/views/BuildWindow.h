#ifndef BUILD_WINDOW_H__
#define BUILD_WINDOW_H__

#include <thread>
#include <vector>
#include <assert.h>

#include "imgui.h"

#include "../EditorClipboard.h"

namespace PhysicsEditor
{
    enum class TargetPlatform
    {
        Windows = 0,
    };

	// Taken from ImGui demo code
	// Usage:
	//  static AppLog my_log;
	//  my_log.AddLog("Hello %d world\n", 123);
	//  my_log.Draw("title");
	struct AppLog
	{
		ImGuiTextBuffer Buf;
		ImGuiTextFilter Filter;
		ImVector<int> LineOffsets; // Index to lines offset. We maintain this with AddLog() calls, allowing us to have a
								   // random access on lines
		bool AutoScroll;
		bool ScrollToBottom;

		AppLog()
		{
			AutoScroll = true;
			ScrollToBottom = false;
			Clear();
		}

		void Clear()
		{
			Buf.clear();
			LineOffsets.clear();
			LineOffsets.push_back(0);
		}

		void AddLog(const char* fmt, ...) IM_FMTARGS(2)
		{
			int old_size = Buf.size();
			va_list args;
			va_start(args, fmt);
			Buf.appendfv(fmt, args);
			va_end(args);
			for (int new_size = Buf.size(); old_size < new_size; old_size++)
				if (Buf[old_size] == '\n')
					LineOffsets.push_back(old_size + 1);
			if (AutoScroll)
				ScrollToBottom = true;
		}

		void Draw(const char* title, const ImVec2& size = ImVec2(0, 0), bool border = false)
		{
			ImGui::BeginChild(title, size, border);

			ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0, 0));
			const char* buf = Buf.begin();
			const char* buf_end = Buf.end();

			// The simplest and easy way to display the entire buffer:
			//   ImGui::TextUnformatted(buf_begin, buf_end);
			// And it'll just work. TextUnformatted() has specialization for large blob of text and will fast-forward to
			// skip non-visible lines. Here we instead demonstrate using the clipper to only process lines that are
			// within the visible area. If you have tens of thousands of items and their processing cost is
			// non-negligible, coarse clipping them on your side is recommended. Using ImGuiListClipper requires A)
			// random access into your data, and B) items all being the  same height, both of which we can handle since
			// we an array pointing to the beginning of each line of text. When using the filter (in the block of code
			// above) we don't have random access into the data to display anymore, which is why we don't use the
			// clipper. Storing or skimming through the search result would make it possible (and would be recommended
			// if you want to search through tens of thousands of entries)
			ImGuiListClipper clipper;
			clipper.Begin(LineOffsets.Size);
			while (clipper.Step())
			{
				for (int line_no = clipper.DisplayStart; line_no < clipper.DisplayEnd; line_no++)
				{
					const char* line_start = buf + LineOffsets[line_no];
					const char* line_end =
						(line_no + 1 < LineOffsets.Size) ? (buf + LineOffsets[line_no + 1] - 1) : buf_end;

					bool pop_color = false;
					if (strncmp(line_start, "[Info]", 6) == 0)
					{
						ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.4f, 1.0f, 0.4f, 1.0f));
						pop_color = true;
					}
					else if (strncmp(line_start, "[Warn]", 6) == 0)
					{
						ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 0.4f, 1.0f));
						pop_color = true;
					}
					else if (strncmp(line_start, "[Error]", 7) == 0)
					{
						ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.4f, 0.4f, 1.0f));
						pop_color = true;
					}
					ImGui::TextUnformatted(line_start, line_end);

					if (pop_color)
						ImGui::PopStyleColor();
				}
			}
			clipper.End();

			ImGui::PopStyleVar();

			if (ScrollToBottom)
				ImGui::SetScrollHereY(1.0f);
			ScrollToBottom = false;

			ImGui::EndChild();
		}
	};

class BuildWindow
{
private:
    std::string mName;
    float mX;
    float mY;
    float mWidth;
    float mHeight;
    bool mOpen;

    TargetPlatform mTargetPlatform;
    AppLog mBuildLog;
     
    float mBuildCompletion;
    std::string mBuildStep;
	std::string mSelectedFolder;

    std::atomic<bool> mLaunchBuild{ false };
    std::atomic<bool> mBuildInProgress{ false };
    std::atomic<bool> mBuildComplete{ false };
    std::thread mBuildWorker;

  public:
    BuildWindow();
    BuildWindow(const std::string& name, float x, float y, float width, float height);
    ~BuildWindow();
    BuildWindow(const BuildWindow &other) = delete;
    BuildWindow &operator=(const BuildWindow &other) = delete;

	void init(Clipboard& clipboard);
    void update(Clipboard& clipboard, bool isOpenedThisFrame);

  private:
    void build();
    void doWork();

};
} // namespace PhysicsEditor

#endif
