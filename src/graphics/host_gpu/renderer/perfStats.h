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

using Clock = std::chrono::steady_clock;

struct State {
	std::array<std::atomic<uint64_t>, static_cast<size_t>(Stage::Count)> nanoseconds {};
	std::atomic<uint64_t>                                                draws {0};
	std::atomic<uint64_t>                                                dispatches {0};
	std::atomic<int64_t>                                                 window_start {0};
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
	std::printf("Perf per second: %.0f draws, %.0f dispatches | resource walk %.0f ms | variant "
	            "lookup %.0f ms | buffers %.0f ms | images %.0f ms | bindings %.0f ms | readback "
	            "waits %.0f ms\n",
	            draws, dispatches, ms[0], ms[1], ms[2], ms[3], ms[4], ms[5]);
	std::fflush(stdout);
}

} // namespace Libs::Graphics::PerfStats

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_HOST_GPU_RENDERER_PERFSTATS_H_ */
