#include "inst_basic_block_trace_pass.h"

#include <initializer_list>
#include <memory>
#include <stdexcept>

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
       ptrStorageBufferUint =
           this->getPtrStorageBufferRuntimeArrayUIntTypeId()](Function* fp) {
        auto* constMgr = context()->get_constant_mgr();
        auto* typeMgr = context()->get_type_mgr();

        // the offset for .counter member
        uint32_t UIntConstZeroId = constMgr->GetUIntConstId(0);
        // used for "+1"
        uint32_t UIntConstOneId = constMgr->GetUIntConstId(1);
        uint32_t tyUint = typeMgr->GetUIntTypeId();

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
          uint32_t counterValId = TakeNextId();
          uint32_t counterIncValId = TakeNextId();

          std::vector<std::unique_ptr<Instruction>> traceInsts;
          traceInsts.push_back(std::make_unique<Instruction>(
                  context(), spv::Op::OpAccessChain,
                  /* ty_id */ ptrStorageBufferUint,
                  /* result_id */ counterPointerId,
                  std::initializer_list<Operand>{
                      {SPV_OPERAND_TYPE_ID, {bbTraceBufferId}},
                      {SPV_OPERAND_TYPE_ID, {UIntConstZeroId}},
                      {SPV_OPERAND_TYPE_ID, {bbTraceIdxConstId}}}));
          traceInsts.push_back(std::make_unique<Instruction>(
                  context(), spv::Op::OpLoad, /* ty_id */ tyUint,
                  /* result_id */ counterValId,
                  std::initializer_list<Operand>{
                      {SPV_OPERAND_TYPE_ID, {counterPointerId}}}));
          traceInsts.push_back(std::make_unique<Instruction>(
                  context(), spv::Op::OpIAdd, /* ty_id */ tyUint,
                  /* result_id */ counterIncValId,
                  std::initializer_list<Operand>{
                      {SPV_OPERAND_TYPE_ID, {counterValId}},
                      {SPV_OPERAND_TYPE_ID, {UIntConstOneId}}}));
          traceInsts.push_back(std::make_unique<Instruction>(
                  context(), spv::Op::OpStore, /* ty_id */ 0,
                  /* result_id */ 0,
                  std::initializer_list<Operand>{
                      {SPV_OPERAND_TYPE_ID, {counterPointerId}},
                      {SPV_OPERAND_TYPE_ID, {counterIncValId}}}));

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

  analysis::Integer tyUint(32, false);


  analysis::RuntimeArray tyRuntimeArray(&tyUint);
  
  // annotate uint offset
  tyRuntimeArray.AddDecoration({6, 4});

  analysis::Struct tyStruct({&tyRuntimeArray});

  uint32_t traceBufferTypeId = typeMgr->GetTypeInstruction(&tyStruct);
  assert(traceBufferTypeId != 0);
  assert(context()->get_def_use_mgr()->NumUses(traceBufferTypeId) == 0 &&
         "used struct type returned");

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

  // TODO: figure out why is it useful
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

uint32_t InstBasicBlockTracePass::getPtrStorageBufferRuntimeArrayUIntTypeId() {
  if (ptrStorageBufferRuntimeArrayUIntTypeId != 0) {
    return ptrStorageBufferRuntimeArrayUIntTypeId;
  }

  auto* typeMgr = context()->get_type_mgr();
  uint32_t resultId = typeMgr->FindPointerToType(
      typeMgr->GetUIntTypeId(), spv::StorageClass::StorageBuffer);
  assert(resultId != 0 && "Could not create desired pointer type");

  ptrStorageBufferRuntimeArrayUIntTypeId = resultId;
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