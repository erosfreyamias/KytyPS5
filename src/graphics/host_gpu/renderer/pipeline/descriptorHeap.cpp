#include "graphics/host_gpu/renderer/pipeline/descriptorHeap.h"

#include "common/assert.h"
#include "common/profiler.h"
#include "graphics/host_gpu/graphicContext.h"
#include "graphics/host_gpu/renderer/masterSemaphore.h"

namespace Libs::Graphics {
namespace {

constexpr uint32_t   DescriptorHeapCount = 1024;
// Bindless stages bind thousands of images per set (PPSA03671: up to 6,418). Pools sized for a
// handful of sets were exhausted almost every draw, which recreated or reset a pool and dropped
// every cached set.
constexpr std::array DescriptorPoolSizes = {
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageBuffer, 16384},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampledImage, 65536},
    vk::DescriptorPoolSize {vk::DescriptorType::eStorageImage, 4096},
    vk::DescriptorPoolSize {vk::DescriptorType::eSampler, 4096},
};

} // namespace

DescriptorHeap::DescriptorHeap(GraphicContext& graphics, MasterSemaphore& master_semaphore)
    : m_graphics(graphics), m_master_semaphore(master_semaphore) {
	CreateDescriptorPool();
}

DescriptorHeap::~DescriptorHeap() {
	m_graphics.device.destroyDescriptorPool(m_current_pool, nullptr);
	for (const auto& [pool, tick]: m_pending_pools) {
		m_master_semaphore.Wait(tick);
		m_graphics.device.destroyDescriptorPool(pool, nullptr);
	}
}

vk::DescriptorSet DescriptorHeap::Commit(vk::DescriptorSetLayout layout) {
	KYTY_PROFILER_FUNCTION();
	EXIT_IF(layout == nullptr);

	auto& batch = m_sets[layout];
	if (batch.size != 0) {
		return batch.sets[--batch.size];
	}
	if (Allocate(layout, batch)) {
		return batch.sets[--batch.size];
	}

	m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
	if (const auto& [pool, tick] = m_pending_pools.front(); m_master_semaphore.IsFree(tick)) {
		m_current_pool = pool;
		m_pending_pools.pop_front();
		EXIT_IF(m_graphics.device.resetDescriptorPool(m_current_pool, {}) != vk::Result::eSuccess);
	} else {
		CreateDescriptorPool();
	}

	m_sets.clear();
	auto& fresh_batch = m_sets[layout];
	// A set bigger than a whole default pool gets a pool scaled to fit it.
	for (uint32_t scale = 2; !Allocate(layout, fresh_batch); scale *= 2) {
		EXIT_IF(scale > 64);
		m_pending_pools.emplace_back(m_current_pool, m_master_semaphore.CurrentTick());
		CreateDescriptorPool(scale);
		fresh_batch = {};
	}
	return fresh_batch.sets[--fresh_batch.size];
}

bool DescriptorHeap::Allocate(vk::DescriptorSetLayout layout, Batch& batch) {
	std::array<vk::DescriptorSetLayout, DescriptorSetBatch> layouts;
	layouts.fill(layout);

	vk::DescriptorSetAllocateInfo allocate {};
	allocate.descriptorPool = m_current_pool;
	allocate.pSetLayouts    = layouts.data();

	for (;;) {
		allocate.descriptorSetCount = batch.allocation;
		const auto result = m_graphics.device.allocateDescriptorSets(&allocate, batch.sets.data());
		if (result == vk::Result::eSuccess) {
			batch.size = batch.allocation;
			return true;
		}
		EXIT_IF(result != vk::Result::eErrorOutOfPoolMemory &&
		        result != vk::Result::eErrorFragmentedPool);
		if (batch.allocation == 1) {
			return false;
		}
		batch.allocation /= 2;
	}
}

void DescriptorHeap::CreateDescriptorPool(uint32_t scale) {
	auto sizes = DescriptorPoolSizes;
	for (auto& size: sizes) {
		size.descriptorCount *= scale;
	}
	vk::DescriptorPoolCreateInfo create {};
	create.maxSets       = DescriptorHeapCount * scale;
	create.poolSizeCount = static_cast<uint32_t>(sizes.size());
	create.pPoolSizes    = sizes.data();
	EXIT_IF(m_graphics.device.createDescriptorPool(&create, nullptr, &m_current_pool) !=
	        vk::Result::eSuccess);
}

} // namespace Libs::Graphics
