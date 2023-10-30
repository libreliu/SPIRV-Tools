#include "inst_basic_block_trace_pass.h"

#include <initializer_list>
#include <memory>
#include <stdexcept>

#include "source/opcode.h"
#include "source/opt/function.h"
#include "source/opt/instruction.h"
#include "source/opt/pass.h"
#include "source/opt/types.h"
#include "source/spirv_constant.h"
#include "spirv-tools/libspirv.h"
#include "spirv/unified1/spirv.hpp11"

namespace spvtools {
namespace opt {

// NOTE: storage buffer slot for basic block counting
//
// ## Debug annotations ##
// OpName %basicBlockTraceBufferType "basicBlockTraceBufferType"
// OpMemberName %basicBlockTraceBufferType 0 "counters"
// OpName %basicBlockTraceBuffer "basicBlockTraceBuffer"
//
// ## Decorations ##
// OpDecorate %_runtimearr_uint ArrayStride 4
// OpMemberDecorate %basicBlockTraceBufferType 0 Offset 0
// OpDecorate %basicBlockTraceBufferType Block
// OpDecorate %basicBlockTraceBuffer DescriptorSet 5
// OpDecorate %basicBlockTraceBuffer Binding 1
//
// ## Type Annotations ##
// %_runtimearr_uint          = OpTypeRuntimeArray %uint
// %basicBlockTraceBufferType = OpTypeStruct %_runtimearr_uint
// %_ptr_StorageBuffer_basicBlockTraceBufferType =
//                  OpTypePointer StorageBuffer %basicBlockTraceBufferType
// %basicBlockTraceBuffer =
//                  OpVariable %_ptr_StorageBuffer_basicBlockTraceBufferType
//                  StorageBuffer
// %_ptr_StorageBuffer_uint = OpTypePointer StorageBuffer %uint
//
// ## Usage ##
// %29 = OpAccessChain %_ptr_StorageBuffer_uint %basicBlockTraceBuffer %int_0
// %int_233 %30 = OpLoad %uint %29 %31 = OpIAdd %uint %30 %uint_1 %32 =
// OpAccessChain %_ptr_StorageBuffer_uint %basicBlockTraceBuffer %int_0 %int_233
// OpStore %32 %31

// Per-invocation version; TODO: fix this
// #version 450
// layout (location = 0) out vec4 outFragColor;
// layout(std430, set = 5, binding = 1) buffer basicBlockTraceBufferType {
//     uint counters[];
// } basicBlockTraceBuffer;
// void main()
// {
//     int index = int(gl_FragCoord.x * 1000);
//     basicBlockTraceBuffer.counters[233] += 1;
// }


// Uni-invocation version
// #version 450
// layout (location = 0) out vec4 outFragColor;
// layout(std430, set = 5, binding = 1) buffer basicBlockTraceBufferType {
//     uint counters[];
// } basicBlockTraceBuffer;
// void main()
// {
//     atomicAdd(basicBlockTraceBuffer.counters[233], 1);
// }

// U64 version
// #version 450
// #extension GL_ARB_gpu_shader_int64 : require
// #extension GL_EXT_shader_atomic_int64 : require
// layout (location = 0) out vec4 outFragColor;
// layout(std430, set = 5, binding = 1) buffer basicBlockTraceBufferType {
//     uint64_t counters[];
// } basicBlockTraceBuffer;
// void main()
// {
//     atomicAdd(basicBlockTraceBuffer.counters[233], 1);
// }

// Differences:
// ## Decorations ##
// OpDecorate %_runtimearr_ulong ArrayStride 8
// OpMemberDecorate %basicBlockTraceBufferType 0 Offset 0
// OpDecorate %basicBlockTraceBufferType Block
// OpDecorate %basicBlockTraceBuffer DescriptorSet 5
// OpDecorate %basicBlockTraceBuffer Binding 1

// ## Type annotations ##
// %ulong = OpTypeInt 64 0
// %_runtimearr_ulong = OpTypeRuntimeArray %ulong
// %basicBlockTraceBufferType = OpTypeStruct %_runtimearr_ulong
// %_ptr_StorageBuffer_ulong = OpTypePointer StorageBuffer %ulong

// ## Function ##
// %15 = OpAccessChain %_ptr_StorageBuffer_ulong %basicBlockTraceBuffer %int_0 %int_233
// %20 = OpAtomicIAdd %ulong %15 %uint_1 %uint_0 %ulong_1

// 1. give all (original) bb a corresponding idx
// 2. instrument
// 2.1 instrument new storage buffer slot
// 2.2 instrument increment code on each beginning of basic block
Pass::Status InstBasicBlockTracePass::Process() {
  labelBasicBlocks();

  // notify
  if (basicBlockCountCallbackFn) {
    basicBlockCountCallbackFn(static_cast<int>(origLabelToTraceIdx.size()));
  }
  
  if (basicBlockCorrespondenceCallbackFn) {
    basicBlockCorrespondenceCallbackFn(&origLabelToTraceIdx);
  }

  // prepare types
  uint32_t bbTraceBufferId = getBasicBlockTraceBufferId();

  // iterate over basic blocks; insert onto the top of the basic block
  ProcessFunction pfn =
      [this, bbTraceBufferId,
       ptrStorageBufferUintOrUlong =
           this->getPtrStorageBufferRuntimeArrayTypeId()](Function* fp) {
        auto* constMgr = context()->get_constant_mgr();
        auto* typeMgr = context()->get_type_mgr();

        // the offset for .counter member
        uint32_t UIntConstZeroId = constMgr->GetUIntConstId(0);
        // used for "+1"
        uint32_t UIntConstOneId = constMgr->GetUIntConstId(1);
        uint32_t tyUint = typeMgr->GetUIntTypeId();
        uint32_t tyUlong = 0;
        if (traceWithU64) {
          analysis::Integer uint64_type(64, false);
          analysis::Type* uint_type = context()->get_type_mgr()->GetRegisteredType(&uint64_type);

          tyUlong = typeMgr->GetTypeInstruction(uint_type);
        }

        uint32_t ULongConstOneId = 0;
        if (traceWithU64) {
          analysis::Integer uint64_type(64, false);

          analysis::Type* uint_type = context()->get_type_mgr()->GetRegisteredType(&uint64_type);
          auto c = constMgr->GetConstant(uint_type, {1, 0});
          ULongConstOneId = constMgr->GetDefiningInstruction(c)->result_id();
        }

        bool changed = false;

        for (auto& bb : *fp) {
          //bb.Dump();
          
          // bb.begin() does not contain OpLabel
          auto bbStartPos = bb.begin();
          if (bbStartPos == bb.end()) {
            // empty bb is not conforming to spir-v specification
            assert(false);
            continue;
          } 
          
          // All OpVariable instructions in a function
          // must be the first instructions in the first block
          while (bbStartPos != bb.end() && bbStartPos->opcode() == spv::Op::OpVariable) {
            ++bbStartPos;
          }

          // This seems legal
          if (bbStartPos == bb.end()) {
            continue;
          }

          changed = true;

          uint32_t bbIdx = bb.id();
          uint32_t bbTraceIdx = this->origLabelToTraceIdx.at(bbIdx);
          uint32_t bbTraceIdxConstId = constMgr->GetUIntConstId(bbTraceIdx);

          uint32_t counterPointerId = TakeNextId();
          // uint32_t counterValId = TakeNextId();
          uint32_t counterIncValId = TakeNextId();

          std::vector<std::unique_ptr<Instruction>> traceInsts;
          traceInsts.push_back(std::make_unique<Instruction>(
                  context(), spv::Op::OpAccessChain,
                  /* ty_id */ ptrStorageBufferUintOrUlong,
                  /* result_id */ counterPointerId,
                  std::initializer_list<Operand>{
                      {SPV_OPERAND_TYPE_ID, {bbTraceBufferId}},
                      {SPV_OPERAND_TYPE_ID, {UIntConstZeroId}},
                      {SPV_OPERAND_TYPE_ID, {bbTraceIdxConstId}}}));

          if (traceWithU64) {
            traceInsts.push_back(std::make_unique<Instruction>(
                context(), spv::Op::OpAtomicIAdd, /* ty_id */ tyUlong,
                /* result_id */ counterIncValId,
                std::initializer_list<Operand>{
                    {SPV_OPERAND_TYPE_ID, {counterPointerId}},
                    /* memory scope id */{SPV_OPERAND_TYPE_SCOPE_ID, {UIntConstOneId}},
                    /* memory semantics id */{SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID, {UIntConstZeroId}},
                    /* val id */{SPV_OPERAND_TYPE_ID, {ULongConstOneId}}}));
          } else {
            traceInsts.push_back(std::make_unique<Instruction>(
                context(), spv::Op::OpAtomicIAdd, /* ty_id */ tyUint,
                /* result_id */ counterIncValId,
                std::initializer_list<Operand>{
                    {SPV_OPERAND_TYPE_ID, {counterPointerId}},
                    /* memory scope id */{SPV_OPERAND_TYPE_SCOPE_ID, {UIntConstOneId}},
                    /* memory semantics id */{SPV_OPERAND_TYPE_MEMORY_SEMANTICS_ID, {UIntConstZeroId}},
                    /* val id */{SPV_OPERAND_TYPE_ID, {UIntConstOneId}}}));
          }

          // traceInsts.push_back(std::make_unique<Instruction>(
          //         context(), spv::Op::OpLoad, /* ty_id */ tyUint,
          //         /* result_id */ counterValId,
          //         std::initializer_list<Operand>{
          //             {SPV_OPERAND_TYPE_ID, {counterPointerId}}}));
          // traceInsts.push_back(std::make_unique<Instruction>(
          //         context(), spv::Op::OpIAdd, /* ty_id */ tyUint,
          //         /* result_id */ counterIncValId,
          //         std::initializer_list<Operand>{
          //             {SPV_OPERAND_TYPE_ID, {counterValId}},
          //             {SPV_OPERAND_TYPE_ID, {UIntConstOneId}}}));
          // traceInsts.push_back(std::make_unique<Instruction>(
          //         context(), spv::Op::OpStore, /* ty_id */ 0,
          //         /* result_id */ 0,
          //         std::initializer_list<Operand>{
          //             {SPV_OPERAND_TYPE_ID, {counterPointerId}},
          //             {SPV_OPERAND_TYPE_ID, {counterIncValId}}}));

          bbStartPos.InsertBefore(std::move(traceInsts));
        }

        return changed;
      };

  bool modified = context()->ProcessEntryPointCallTree(pfn);

  return modified ? Status::SuccessWithChange : Status::SuccessWithoutChange;
}

void InstBasicBlockTracePass::labelBasicBlocks() {
  uint32_t bbIdx = 0;

  // returns whether the process function have modified something
  ProcessFunction pfn = [&bbIdx, &origLabelToTraceIdx =
                                     this->origLabelToTraceIdx](Function* fp) {
    for (auto& bb : *fp) {
      auto labelInst = bb.GetLabelInst();
      assert(labelInst->HasResultId());

      origLabelToTraceIdx[labelInst->result_id()] = bbIdx++;
    }

    return false;
  };

  context()->ProcessEntryPointCallTree(pfn);
}

// Suits Vulkan 1.3
// => Decorations vary between Vulkan 1.0 and Vulkan 1.3
uint32_t InstBasicBlockTracePass::getBasicBlockTraceBufferId() {
  if (basicBlockTraceBufferId != 0) {
    return basicBlockTraceBufferId;
  }

  analysis::DecorationManager* decoMgr = get_decoration_mgr();
  analysis::TypeManager* typeMgr = context()->get_type_mgr();

  std::unique_ptr<analysis::Integer> tyUint;
  if (traceWithU64) {
    tyUint = std::make_unique<analysis::Integer>(64, false);
  } else {
    tyUint = std::make_unique<analysis::Integer>(32, false);
  }

  analysis::RuntimeArray tyRuntimeArray(tyUint.get());
  
  // annotate uint offset
  // 6 - ArrayStride; 4 - the stride value
  if (traceWithU64) {
    tyRuntimeArray.AddDecoration({6, 8});
  } else {
    tyRuntimeArray.AddDecoration({6, 4});
  }

  analysis::Struct tyStruct({&tyRuntimeArray});

  uint32_t traceBufferTypeId = typeMgr->GetTypeInstruction(&tyStruct);
  assert(traceBufferTypeId != 0);

  // This is possible depending on SPIR-V input, so move from assertion
  // to runtime check
  if (context()->get_def_use_mgr()->NumUses(traceBufferTypeId) == 0) {
    // We have runtime exceptions disabled...
    assert("used struct type returned");
  }

  decoMgr->AddDecoration(traceBufferTypeId, uint32_t(spv::Decoration::Block));
  decoMgr->AddMemberDecoration(traceBufferTypeId, 0,
                               uint32_t(spv::Decoration::Offset), 0);

  uint32_t traceBufferPointerTypeId = typeMgr->FindPointerToType(
      traceBufferTypeId, spv::StorageClass::StorageBuffer);

  uint32_t traceBufferId = TakeNextId();
  std::unique_ptr<Instruction> newVarOp(new Instruction(
      context(), spv::Op::OpVariable, traceBufferPointerTypeId, traceBufferId,
      {{spv_operand_type_t::SPV_OPERAND_TYPE_LITERAL_INTEGER,
        {uint32_t(spv::StorageClass::StorageBuffer)}}}));

  context()->AddGlobalValue(std::move(newVarOp));
  context()->AddDebug2Inst(std::make_unique<Instruction>(
      context(), spv::Op::OpName, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {traceBufferTypeId}},
          {SPV_OPERAND_TYPE_LITERAL_STRING,
           utils::MakeVector("BasicBlockTraceBuffer")}}));
  context()->AddDebug2Inst(std::make_unique<Instruction>(
      context(), spv::Op::OpMemberName, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {traceBufferTypeId}},
          {SPV_OPERAND_TYPE_LITERAL_INTEGER, {0}},
          {SPV_OPERAND_TYPE_LITERAL_STRING, utils::MakeVector("counters")}}));
  context()->AddDebug2Inst(std::make_unique<Instruction>(
      context(), spv::Op::OpName, 0, 0,
      std::initializer_list<Operand>{
          {SPV_OPERAND_TYPE_ID, {traceBufferId}},
          {SPV_OPERAND_TYPE_LITERAL_STRING,
           utils::MakeVector("basic_block_trace_buffer")}}));

  decoMgr->AddDecorationVal(traceBufferId,
                            uint32_t(spv::Decoration::DescriptorSet),
                            kTraceBufferDescriptorSet);
  decoMgr->AddDecorationVal(traceBufferId, uint32_t(spv::Decoration::Binding),
                            kTraceBufferBinding);
  addStorageBufferExt();
  if (traceWithU64) {
    // TODO: check if we need
    // - OpSourceExtension "GL_ARB_gpu_shader_int64"
    // - OpSourceExtension "GL_EXT_shader_atomic_int64"
    addInt64Caps();
  }

  // Before version 1.4, the interface’s storage classes are limited to the 
  // Input and Output storage classes. Starting with version 1.4, the 
  // interface’s storage classes are all storage classes used in declaring
  // all global variables referenced by the entry point’s call tree.
  //
  // this is copied from spvtools::opt::InstrumentPass::GetOutputBufferId
  if (get_module()->version() >= SPV_SPIRV_VERSION_WORD(1, 4)) {
    // Add the new buffer to all entry points.
    for (auto& entry : get_module()->entry_points()) {
      entry.AddOperand({SPV_OPERAND_TYPE_ID, {traceBufferId}});
      context()->AnalyzeUses(&entry);
    }
  }

  assert(traceBufferId != 0);
  basicBlockTraceBufferId = traceBufferId;
  return traceBufferId;
}

void InstBasicBlockTracePass::addStorageBufferExt() {
  if (storageBufferExtDefined) return;
  if (!get_feature_mgr()->HasExtension(kSPV_KHR_storage_buffer_storage_class)) {
    context()->AddExtension("SPV_KHR_storage_buffer_storage_class");
  }
  storageBufferExtDefined = true;
}

void InstBasicBlockTracePass::addInt64Caps() {
  if (int64CapsDefined) return;
  if (!get_feature_mgr()->HasCapability(spv::Capability::Int64)) {
    context()->AddCapability(spv::Capability::Int64);
  }
  if (!get_feature_mgr()->HasCapability(spv::Capability::Int64Atomics)) {
    context()->AddCapability(spv::Capability::Int64Atomics);
  }
  int64CapsDefined = true;
}

uint32_t InstBasicBlockTracePass::getPtrStorageBufferRuntimeArrayTypeId() {
  if (ptrStorageBufferRuntimeArrayTypeId != 0) {
    return ptrStorageBufferRuntimeArrayTypeId;
  }

  uint32_t uintTypeId = 0;
  if (traceWithU64) {
    auto* typeMgr = context()->get_type_mgr();

    analysis::Integer uint64_type(64, false);
    analysis::Type *uint64TypeObj = typeMgr->GetRegisteredType(&uint64_type);
    uintTypeId = typeMgr->GetTypeInstruction(uint64TypeObj);
  } else {
    uintTypeId = context()->get_type_mgr()->GetUIntTypeId();
  }

  auto* typeMgr = context()->get_type_mgr();
  uint32_t resultId = typeMgr->FindPointerToType(
      uintTypeId, spv::StorageClass::StorageBuffer);
  assert(resultId != 0 && "Could not create desired pointer type");

  ptrStorageBufferRuntimeArrayTypeId = resultId;
  return resultId;
}

uint32_t InstBasicBlockTracePass::getStride4UIntRuntimeArrayTypeId() {
  if (stride4UIntRuntimeArrayTypeId != 0) {
    return stride4UIntRuntimeArrayTypeId;
  }

  auto *typeMgr = context()->get_type_mgr();

  analysis::Integer tyUint(32, false);
  analysis::RuntimeArray tyRuntimeArray(&tyUint);

  // this is ugly, but just works?
  // - since type comparisons are aware of this
  // See spvtools::opt::TypeManager::AttachDecoration on how things get converted
  // OpDecorate %_runtimearr_uint ArrayStride 4
  tyRuntimeArray.AddDecoration({6, 4});
  
  stride4UIntRuntimeArrayTypeId = typeMgr->GetTypeInstruction(&tyRuntimeArray);
  assert(stride4UIntRuntimeArrayTypeId != 0);
  return stride4UIntRuntimeArrayTypeId;
}

void InstBasicBlockTracePass::registerBasicBlockCountRetrievalCallback(
  std::function<void(int)> callbackFn
) {
  basicBlockCountCallbackFn = callbackFn;
}


void InstBasicBlockTracePass::registerBasicBlockCorrespondenceCallback(
  std::function<void(const std::map<int, int> *)> callbackFn
) {
  basicBlockCorrespondenceCallbackFn = callbackFn;
}

}  // namespace opt
}  // namespace spvtools