#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>

namespace Libs::Graphics::PerfStats {

// Command-thread time per draw stage, grouped like the PPSA03671 render-path profile. Stages can
// overlap (a buffer lookup may wait for a readback), so the totals are approximate.
enum class Stage : uint32_t { ResourceWalk, VariantLookup, Buffers, Images, Bindings, Readback, Count };

// What asked for a GPU-to-CPU buffer readback. Each readback that downloads data waits for the GPU.
enum class Readback : uint32_t {
	CpuRead,
	CpuWrite,
	FileRead,
	ShaderSetup,
	ColorClear,
	IndirectArgs,
	Other,
	Count,
};

using Clock = std::chrono::steady_clock;

struct State {
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Stage::Count)> nanoseconds {};
	std::atomic<uint64_t>                                                draws {0};
	std::atomic<uint64_t>                                                dispatches {0};
	std::atomic<uint64_t>                                                images {0};
	std::atomic<int64_t>                                                 window_start {0};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Readback::Count)> readbacks {};
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Readback::Count)> readback_ns {};
};

inline State& Get() {
	static State state;
	return state;
}

inline int64_t NowNanoseconds() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch())
	    .count();
}

class Scope {
public:
	explicit Scope(Stage stage): m_stage(stage), m_start(NowNanoseconds()) {}
	~Scope() {
		Get().nanoseconds[static_cast<size_t>(m_stage)].fetch_add(
		    static_cast<uint64_t>(NowNanoseconds() - m_start), std::memory_order_relaxed);
	}
	Scope(const Scope&)            = delete;
	Scope& operator=(const Scope&) = delete;

private:
	Stage   m_stage;
	int64_t m_start;
};

// The readback source of the calling thread, set around code that may read GPU-written memory.
inline thread_local Readback t_readback_source = Readback::Other;

class ReadbackSource {
public:
	explicit ReadbackSource(Readback source): m_previous(t_readback_source) {
		t_readback_source = source;
	}
	~ReadbackSource() { t_readback_source = m_previous; }
	ReadbackSource(const ReadbackSource&)            = delete;
	ReadbackSource& operator=(const ReadbackSource&) = delete;

private:
	Readback m_previous;
};

// Records one readback request; only requests that downloaded data are counted per source.
inline void CountReadback(Readback source, bool downloaded, int64_t nanoseconds) {
	auto&      state = Get();
	const auto time  = static_cast<uint64_t>(nanoseconds);
	state.nanoseconds[static_cast<size_t>(Stage::Readback)].fetch_add(time,
	                                                                 std::memory_order_relaxed);
	if (downloaded) {
		state.readbacks[static_cast<size_t>(source)].fetch_add(1, std::memory_order_relaxed);
		state.readback_ns[static_cast<size_t>(source)].fetch_add(time, std::memory_order_relaxed);
	}
}

// Counts the images one draw or dispatch stage binds.
inline void CountImages(size_t count) {
	Get().images.fetch_add(count, std::memory_order_relaxed);
}

// Counts one draw or dispatch; every five seconds prints the per-second time of each stage.
inline void CountWork(bool dispatch) {
	auto& state = Get();
	(dispatch ? state.dispatches : state.draws).fetch_add(1, std::memory_order_relaxed);
	const auto now   = NowNanoseconds();
	auto       start = state.window_start.load(std::memory_order_relaxed);
	if (start == 0) {
		state.window_start.compare_exchange_strong(start, now);
		return;
	}
	constexpr int64_t Window = 5'000'000'000;
	if (now - start < Window || !state.window_start.compare_exchange_strong(start, now)) {
		return;
	}
	const auto seconds = static_cast<double>(now - start) / 1e9;
	std::array<double, static_cast<size_t>(Stage::Count)> ms {};
	for (size_t stage = 0; stage < ms.size(); ++stage) {
		ms[stage] = static_cast<double>(state.nanoseconds[stage].exchange(0)) / 1e6 / seconds;
	}
	const auto draws      = static_cast<double>(state.draws.exchange(0)) / seconds;
	const auto dispatches = static_cast<double>(state.dispatches.exchange(0)) / seconds;
	const auto images     = static_cast<double>(state.images.exchange(0)) / seconds;
	std::array<double, static_cast<size_t>(Readback::Count)> count {};
	std::array<double, static_cast<size_t>(Readback::Count)> wait {};
	for (size_t source = 0; source < count.size(); ++source) {
		count[source] = static_cast<double>(state.readbacks[source].exchange(0)) / seconds;
		wait[source]  = static_cast<double>(state.readback_ns[source].exchange(0)) / 1e6 / seconds;
	}
	std::printf("Perf per second: %.0f draws, %.0f dispatches, %.0f image bindings | resource walk "
	            "%.0f ms | variant lookup %.0f ms | buffers %.0f ms | images %.0f ms | bindings "
	            "%.0f ms | readback waits %.0f ms\n",
	            draws, dispatches, images, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5]);
	std::printf("Perf per second (readbacks): cpu read %.0f (%.0f ms) | cpu write %.0f (%.0f ms) | "
	            "file read %.0f (%.0f ms) | shader setup %.0f (%.0f ms) | color clear %.0f (%.0f "
	            "ms) | indirect args %.0f (%.0f ms) | other %.0f (%.0f ms)\n",
	            count[0], wait[0], count[1], wait[1], count[2], wait[2], count[3], wait[3],
	            count[4], wait[4], count[5], wait[5], count[6], wait[6]);
	std::fflush(stdout);
}

} // namespace Libs::Graphics::PerfStats

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_ */
