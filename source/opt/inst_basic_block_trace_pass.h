#pragma once

#include <list>
#include <memory>
#include <vector>

#include "source/opt/instruction.h"
#include "source/opt/ir_builder.h"
#include "source/opt/pass.h"

// This class does not use the instrument interface for khronos validation
// layer since its so tightly coupled, and fits more in a stream-write
// fashion, however in this pass every address could be preallocated.

namespace spvtools {

namespace opt {

class InstBasicBlockTracePass : public Pass {
 public:
  const char* name() const override { return "inst-basic-block-trace"; }
  Status Process() override;

  // TODO: figure out what can be preserved
  IRContext::Analysis GetPreservedAnalyses() override {
    return IRContext::kAnalysisNone;
  }

  void registerBasicBlockCountRetrievalCallback(
      std::function<void(int)> callbackFn);

  // map from "result id of OpLabel" to basic block trace idx
  void registerBasicBlockCorrespondenceCallback(
      std::function<void(const std::map<int, int> *)> callbackFn);

 private:
  void labelBasicBlocks();
  uint32_t getBasicBlockTraceBufferId();
  uint32_t getPtrStorageBufferRuntimeArrayUIntTypeId();

  const int kTraceBufferDescriptorSet = 5;
  const int kTraceBufferBinding = 1;
  bool storageBufferExtDefined = false;

  void addStorageBufferExt();

  std::map<int, int> origLabelToTraceIdx;

  std::function<void(int)> basicBlockCountCallbackFn;
  std::function<void(const std::map<int, int> *)> basicBlockCorrespondenceCallbackFn;

  uint32_t basicBlockTraceBufferId = 0;
  uint32_t ptrStorageBufferRuntimeArrayUIntTypeId = 0;
};

}  // namespace opt
}  // namespace spvtools