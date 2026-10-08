#ifndef EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
#define EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_

#include "common/assert.h"
#include "graphics/host_gpu/renderer/cache/bufferCache.h"
#include "graphics/host_gpu/renderer/cache/textureCache.h"
#include "graphics/host_gpu/renderer/image/image.h"
#include "graphics/host_gpu/vulkanCommon.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderBindings.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace Libs::Graphics {

struct ShaderStageRuntime;

struct TextureBinding {
	ImageId                    image_id;
	vk::ImageView              image_view = nullptr;
	TextureCache::ImageDesc    desc;
	vk::ImageLayout            layout = vk::ImageLayout::eUndefined;
	std::vector<vk::ImageView> mip_views;
};

// A texture descriptor and the shader's use of it, which decide its guest image description.
struct TextureDescKey {
	std::array<uint32_t, 8> dwords {};
	std::array<uint32_t, 4> usage {};

	bool operator==(const TextureDescKey& other) const = default;
};

struct TextureDescKeyHash {
	size_t operator()(const TextureDescKey& key) const noexcept;
};

// The part of a texture lookup that depends only on its TextureDescKey.
struct DecodedTextureDesc {
	TextureCache::ImageDesc desc;
	vk::Format              pixel_format      = vk::Format::eUndefined;
	vk::Format              view_format       = vk::Format::eUndefined;
	uint64_t                size              = 0;
	bool                    null_texture      = false;
	bool                    shader_conversion = false;
};

struct PreparedBindings {
	struct BufferSource {
		uint64_t address = 0;
		uint64_t size    = 0;
		BufferId id;
	};

	// The draw owns the immutable compiled-program/runtime-snapshot association through commit.
	const ShaderStageRuntime* runtime = nullptr;
	// Keep the resolved guest range through cache preparation; only the host buffer ID may
	// become stale and need resolving again when bindings are rebound.
	std::vector<BufferSource>             buffer_sources;
	std::vector<vk::DescriptorBufferInfo> buffers;
	std::vector<TextureBinding>           images;
	std::vector<vk::Sampler>              samplers;
	vk::DescriptorBufferInfo              gds {nullptr, 0, VK_WHOLE_SIZE};
	vk::DescriptorBufferInfo              flattened_srt;
	vk::DescriptorBufferInfo              shader_data_buffer;
	vk::DescriptorBufferInfo              shared_memory;
	std::vector<uint32_t>                 shader_data;

	// The sampled textures in `images` stay acquired after a draw. A later draw of the same
	// program reuses each one whose descriptor and image are unchanged.
	struct TextureReuse {
		struct Stage {
			std::vector<TextureBinding>                        images;
			std::vector<ShaderRecompiler::IR::DescriptorValue> descriptors;
			std::vector<ImageId>                               ids;
			std::vector<uint64_t>                              layout_versions;
		};

		// The program `images` belongs to (CompiledShaderInfo::serial). Per binding: its
		// descriptor, and its image with that image's layout version when acquired; the ID is
		// invalid for one that cannot be reused.
		uint64_t                                           program = 0;
		std::vector<ShaderRecompiler::IR::DescriptorValue> descriptors;
		std::vector<ImageId>                               ids;
		std::vector<uint64_t>                              layout_versions;
		// Set once every view in `images` is acquired; cleared while they are re-resolved.
		bool valid = false;
		// Per binding, whether this draw reuses it.
		std::vector<uint8_t> reused;
		// The textures of other programs these bindings drew recently, by program.
		std::unordered_map<uint64_t, Stage> stages;
	} texture_reuse;
};

[[nodiscard]] vk::DescriptorType
NativeDescriptorType(ShaderRecompiler::IR::DescriptorBindingKind kind);
[[nodiscard]] uint32_t
NativeDescriptorCount(const ShaderRecompiler::IR::DescriptorBinding& binding);
[[nodiscard]] vk::DescriptorImageInfo MakeImageInfo(const TextureBinding& texture,
                                                    uint32_t              element = 0);

template <typename T>
[[nodiscard]] T DecodeNativeDescriptor(const ShaderRecompiler::IR::DescriptorValue& value) {
	static_assert(std::is_trivially_copyable_v<T>);
	static_assert(sizeof(T) % sizeof(uint32_t) == 0);
	T result {};
	EXIT_IF(value.dword_count < sizeof(result) / sizeof(uint32_t));
	std::memcpy(&result, value.dwords.data(), sizeof(result));
	return result;
}

[[nodiscard]] bool IsSupportedDepthTextureEncoding(const ShaderTextureResource& descriptor,
                                                   bool r128 = false);
void ValidateStorageTexture(const ShaderRecompiler::IR::ImageResource& resource,
                            const ShaderTextureResource& descriptor, uint64_t size);

} // namespace Libs::Graphics

#endif // EMULATOR_SRC_GRAPHICS_HOST_GPU_RENDERER_DESCRIPTORS_H_
