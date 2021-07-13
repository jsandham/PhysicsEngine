#ifndef SPRITERENDERER_H__
#define SPRITERENDERER_H__

#include <vector>

#include "Component.h"

#include "../core/Color.h"

namespace PhysicsEngine
{
	class SpriteRenderer : public Component
	{
	private:
		Guid mSpriteId;

	public:
		Color mColor;
		bool mSpriteChanged;
		bool mIsStatic;
		bool mEnabled;

	public:
		SpriteRenderer();
		SpriteRenderer(Guid id);
		~SpriteRenderer();

		virtual void serialize(YAML::Node& out) const override;
		virtual void deserialize(const YAML::Node& in) override;

		virtual int getType() const override;
		virtual std::string getObjectName() const override;

		void setSprite(Guid id);
		Guid getSprite() const;

		//void init();
		//void drawSprite(Camera* camera, GLint texture, const glm::vec2& position, const glm::vec2& size, float rotate = 0.0f);
	};

	template <> struct ComponentType<SpriteRenderer>
	{
		static constexpr int type = PhysicsEngine::SPRITERENDERER_TYPE;
	};

	template <> struct IsComponentInternal<SpriteRenderer>
	{
		static constexpr bool value = true;
	};
}

#endif
