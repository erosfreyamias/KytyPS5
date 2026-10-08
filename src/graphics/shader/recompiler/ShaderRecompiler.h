#ifndef EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_
#define EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_

#include "common/common.h"
#include "common/stringUtils.h"
#include "graphics/shader/recompiler/ir/passes/ResourceMaterialization.h"
#include "graphics/shader/shader.h"

#include <span>
#include <vector>

namespace Libs::Graphics::ShaderRecompiler {

// A function call (S_SWAPPC_B64) whose callee address the shader reads from user SGPRs.
struct ShaderCallSite {
	uint32_t pc        = 0; // byte offset of the S_SWAPPC_B64
	uint32_t user_sgpr = 0; // the callee address is user_data[user_sgpr..user_sgpr + 1]
};

// The code a call site resolved to for one dispatch, through the callee's last return.
struct ShaderCallee {
	uint32_t                  pc = 0;
	std::span<const uint32_t> code;
};

// Finds the function calls in a shader; empty when it makes none. Aborts when a call target
// does not come from user SGPRs.
[[nodiscard]] std::vector<ShaderCallSite> FindShaderCalls(std::span<const uint32_t> code);
// Returns the number of words of the function at the start of code, through its last return,
// or 0 when no return is found within code.
[[nodiscard]] uint32_t MeasureShaderFunction(std::span<const uint32_t> code);

struct CompileOptions {
	ShaderType                  stage           = ShaderType::Compute;
	uint32_t                    wave_size       = 64;
	uint32_t                    user_data_base  = 0;
	uint64_t                    shader_hash     = 0;
	bool                        dump_ir                    = true;
	bool                        early_dump                 = false;
	const char*                 dump_label                 = nullptr;
	std::span<const uint32_t>   user_data;
	std::span<const uint32_t>   back_code;
	std::span<const ShaderCallee> callees;
	ShaderStageInputInfo        input_info;
};

struct TranslateResult {
	IR::Program program;
	std::string decoded_dump;
	std::string cfg_dump;
};

struct CompileResult {
	std::vector<uint32_t>  spirv;
	std::string            decoded_dump;
	std::string            ir_dump;
	IR::Program            program;
};

[[nodiscard]] TranslateResult TranslateProgram(std::span<const uint32_t> code,
                                               const CompileOptions& options);
[[nodiscard]] CompileResult CompileProgram(TranslateResult translated,
                                           const CompileOptions& options,
                                           const IR::ResourceSpecialization& specialization,
	                                       uint32_t push_data_start_dword = 0);

} // namespace Libs::Graphics::ShaderRecompiler

#endif /* EMULATOR_INCLUDE_EMULATOR_GRAPHICS_SHADER_RECOMPILER_SHADERRECOMPILER_H_ */
