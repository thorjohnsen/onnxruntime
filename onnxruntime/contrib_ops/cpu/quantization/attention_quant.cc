// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "attention_quant.h"
#include "contrib_ops/cpu/bert/attention_helper.h"
#include "core/providers/common.h"
#include "core/util/math.h"
#include "core/util/qmath.h"
#include "core/util/math_cpuonly.h"
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/mlas/inc/mlas.h"


#include "core/providers/cpu/math/matmul_helper.h"
#include "core/util/qmath.h"
#include "core/providers/common.h"


using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace contrib {
// These ops are internal-only, so register outside of onnx
#define REGISTER_KERNEL_TYPED(T, QInput, QWeight)                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      QAttention,                                                        \
      kMSDomain,                                                         \
      1,                                                                 \
      T##_##QInput##_##QWeight,                                          \
      kCpuExecutionProvider,                                             \
      KernelDefBuilder()                                                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<QInput>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<QWeight>())  \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()), \
      QAttention<T, QInput, QWeight>);

REGISTER_KERNEL_TYPED(float, uint8_t, int8_t)
REGISTER_KERNEL_TYPED(float, uint8_t, uint8_t)

template <typename T, typename QInput, typename QWeight>
QAttention<T, QInput, QWeight>::QAttention(const OpKernelInfo& info) : OpKernel(info), AttentionBase(info) {
}

template <typename T, typename QInput, typename QWeight>
Status QAttention<T, QInput, QWeight>::Compute(OpKernelContext* context) const {
  // Input and output shapes:
  //   Input 0 - input             : (batch_size, sequence_length, hidden_size)
  //   Input 1 - weights           : (hidden_size, 3 * hidden_size)
  //   Input 2 - bias              : (3 * hidden_size)
  //   Input 3 - input_scale       : scalar
  //   Input 4 - weight_scale      : scalar
  //   Input 5 - mask_index        : (batch_size)
  //   Input 6 - input_zero_point  : scalar
  //   Input 7 - weight_zero_point : scalar
  //   Output                      : (batch_size, sequence_length, hidden_size)
  //   ORT_RETURN_IF_ERROR(CheckInputs(context));
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* weights = context->Input<Tensor>(1);
  const Tensor* bias = context->Input<Tensor>(2);
  const Tensor* input_scale_tensor = context->Input<Tensor>(3);
  const Tensor* weight_scale_tensor = context->Input<Tensor>(4);
  const Tensor* mask_index = context->Input<Tensor>(5);
  const Tensor* i_zp_tensor = context->Input<Tensor>(6);
  const Tensor* w_zp_tensor = context->Input<Tensor>(7);

  ORT_RETURN_IF_ERROR(AttentionBase::CheckInputs(input, weights, bias, mask_index));

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(input_scale_tensor),
                    "input scale must be a scalar or 1D tensor of size 1");
  T input_scale = *(input_scale_tensor->template Data<T>());

  ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(weight_scale_tensor),
                    "weight must be a scalar or 1D tensor of size 1");
  T weight_scale = *(weight_scale_tensor->template Data<T>());

  T dequant_scale = input_scale * weight_scale;

  QInput input_zero_point = 0;
  if (i_zp_tensor != nullptr) {
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(i_zp_tensor),
                      "input zero point must be a scalar or 1D tensor of size 1.");
    input_zero_point = *i_zp_tensor->template Data<QInput>();
  }

  QWeight weight_zero_point = 0;
  if (w_zp_tensor != nullptr) {
    // CUDA only support symmetric quantization for Attention
    ORT_RETURN_IF_NOT(IsScalarOr1ElementVector(w_zp_tensor),
                      "weight zero point must be a scalar or 1D tensor of size 1.");
    weight_zero_point = *w_zp_tensor->template Data<QWeight>();
  }

  const auto dims = input->Shape().GetDims();
  const int batch_size = static_cast<int>(dims[0]);
  const int sequence_length = static_cast<int>(dims[1]);
  const int hidden_size = static_cast<int>(dims[2]);
  const int head_size = hidden_size / num_heads_;

  TensorShape output_shape(dims);
  Tensor* output = context->Output(0, output_shape);

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  constexpr size_t element_size = sizeof(T);

  auto* tp = context->GetOperatorThreadPool();
  // STEP.1: gemm_data(BS, 3NH) = Scale(input(BS, NH) x weights(NH, 3NH)) + bias(3NH)
  auto gemm_data = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * element_size);
  BufferUniquePtr gemm_buffer(gemm_data, BufferDeleter(allocator));

  auto Q = reinterpret_cast<T*>(gemm_data);
  auto K = Q + batch_size * sequence_length * hidden_size;
  auto V = K + batch_size * sequence_length * hidden_size;
  T* QKV[3] = {Q, K, V};

#if 0
  auto gemm_data_quant = allocator->Alloc(SafeInt<size_t>(batch_size) * sequence_length * 3 * hidden_size * sizeof(int32_t));
  BufferUniquePtr gemm_buffer_quant(gemm_data_quant, BufferDeleter(allocator));
#else
  auto gemm_data_quant = gemm_data; // inplace
#endif

  auto Q_quant = reinterpret_cast<int32_t*>(gemm_data_quant);
  auto K_quant = Q_quant + batch_size * sequence_length * hidden_size;
  auto V_quant = K_quant + batch_size * sequence_length * hidden_size;
  int32_t* QKV_quant[3] = {Q_quant, K_quant, V_quant};

  {
    const int loop_len = 3 * batch_size * num_heads_;
    const auto input_data = input->template Data<QInput>();
    const auto weights_data = weights->template Data<QWeight>();
    const auto bias_data = bias->template Data<T>();

    const double cost =
        static_cast<double>(sequence_length) * static_cast<double>(head_size) * static_cast<double>(hidden_size);
    ThreadPool::TryParallelFor(tp, loop_len, cost, [&](std::ptrdiff_t begin, std::ptrdiff_t end) {
      for (std::ptrdiff_t i = begin; i != end; ++i) {
        const int batch_index = static_cast<int>((i / 3) / num_heads_);
        const int head_index = static_cast<int>((i / 3) % num_heads_);
        const int qkv_index = static_cast<int>(i % 3);

        int input_offset = batch_index * sequence_length * hidden_size;
        int weights_offset = qkv_index * hidden_size + head_index * head_size;
//        int32_t* qkv_dest = QKV_quant[qkv_index];
        int qkv_offset = (batch_index * num_heads_ + head_index) * (sequence_length * head_size);

        float* qkv_dest_fp32 = QKV[qkv_index];

        MLAS_GEMM_U8X8_PARAMETERS gemm_parameters = { };

        gemm_parameters.M = sequence_length;
        gemm_parameters.N = head_size;
        gemm_parameters.K = hidden_size;
        gemm_parameters.A = input_data + input_offset;
        gemm_parameters.lda = hidden_size;
        gemm_parameters.offa = input_zero_point;
        gemm_parameters.B = (const uint8_t*)weights_data + weights_offset;
        gemm_parameters.ldb = 3 * hidden_size;
        gemm_parameters.offb = weight_zero_point;
        gemm_parameters.BTypeIsSigned = std::is_signed<QWeight>::value;
        gemm_parameters.C = (int32_t*)qkv_dest_fp32 + qkv_offset;
        gemm_parameters.ldc = head_size;
        gemm_parameters.CTypeIsFloat = true;
        gemm_parameters.Multiplier = &dequant_scale;
        gemm_parameters.Bias = bias_data + weights_offset;

        MlasGemm(&gemm_parameters, nullptr);

#if 0
        //                   original           transposed            iteration
        // A: input          (BxSxNxH)          (B.)S x NH            S x NH
        // B: weights        (NxHx3xNxH)        NH  x (3.N.)H         NH x H
        // C: QKV[qkv_index] (3xBxNxSxH)        (3.B.N.)S x H         S x H
        QGemm(sequence_length,                // M      = S
              head_size,                      // N      = H
              hidden_size,                    // K      = NH
              input_data + input_offset,      // A
              hidden_size,                    // lda    = NH
              input_zero_point,               // input zero point
              weights_data + weights_offset,  // B
              3 * hidden_size,                // ldb    = 3NH
              weight_zero_point,              // weight zero point
              qkv_dest + qkv_offset,          // C
              head_size,                      // ldc
              nullptr                         // use single-thread
              );
#endif

        // dequantize and add bias
        // broadcast 3NH -> (3.B.N.S.H)
        const T* bias_src = bias_data + weights_offset;
//        int32_t* gemm_quant_src = QKV_quant[qkv_index] + qkv_offset;
        T* data_dest = QKV[qkv_index] + qkv_offset;
        for (int seq_index = 0; seq_index < sequence_length; seq_index++) {
          for (int head_idx = 0; head_idx < head_size; head_idx++) {
            data_dest[head_idx] = data_dest[head_idx] + bias_src[head_idx];
          }
          data_dest += head_size;
        }
      }
    });
  }

  // STEP.2: compute the attention score. It does 2 things:
  //         I. attention_probs(B, N, S, S) = 1/sqrt(H) x Q(B, N, S, H) x K'(B, N, S, H -> B, N, H, S) +
  //                                         1 x mask_data(B, N, S, S)
  //         II.attention_probs(B, N, S, S) = Softmax(attention_probs)
  size_t attention_probs_bytes = SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * sequence_length * element_size;
  auto attention_probs = allocator->Alloc(attention_probs_bytes);
  BufferUniquePtr scratch_buffer(attention_probs, BufferDeleter(allocator));

  size_t mask_data_bytes = 0;
  if (mask_index != nullptr) {
    mask_data_bytes = SafeInt<size_t>(batch_size) * sequence_length * sequence_length * element_size;
  } else if (is_unidirectional_) {
    mask_data_bytes = SafeInt<size_t>(sequence_length) * sequence_length * element_size;
  }

  void* mask_data = nullptr;
  if (mask_data_bytes > 0) {
    mask_data = allocator->Alloc(mask_data_bytes);
    memset(mask_data, 0, mask_data_bytes);
  }
  BufferUniquePtr mask_data_buffer(mask_data, BufferDeleter(allocator));

  const int32_t* mask_index_data = mask_index != nullptr ? mask_index->template Data<int32_t>() : nullptr;

  ComputeAttentionProbs<T>(static_cast<T*>(attention_probs), Q, K, mask_index_data, static_cast<T*>(mask_data),
                           batch_size, sequence_length, head_size, num_heads_, is_unidirectional_, tp);

  // STEP.3: compute the attentionScore * Value. It does: out_tmp(B, N, S, H) = attention_probs(B, N, S, S) x V(B, N, S, H)
  auto out_tmp_data =
      allocator->Alloc(SafeInt<size_t>(batch_size) * num_heads_ * sequence_length * head_size * element_size);
  BufferUniquePtr out_tmp_buffer(out_tmp_data, BufferDeleter(allocator));

  ComputeVxAttentionScore(output->template MutableData<T>(), static_cast<T*>(out_tmp_data), static_cast<T*>(attention_probs), V,
                          batch_size, sequence_length, head_size, num_heads_, hidden_size, tp);

  return Status::OK();
}

template <typename T1, typename T2>
class MegaFusion final : public OpKernel {
 public:
  MegaFusion(const OpKernelInfo& info) : OpKernel(info) {
    has_a_zero_point_ = false;
    has_b_zero_point_ = false;
    if (info.GetInputCount() > 2) {
      has_a_zero_point_ = true;
    }
    if (info.GetInputCount() > 3) {
      has_b_zero_point_ = true;
    }
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  bool has_a_zero_point_;
  bool has_b_zero_point_;
};

template <typename T1, typename T2>
Status MegaFusion<T1, T2>::Compute(OpKernelContext* ctx) const {
  concurrency::ThreadPool* thread_pool = ctx->GetOperatorThreadPool();

  auto a = ctx->Input<Tensor>(0);
  auto b = ctx->Input<Tensor>(1);
  ORT_ENFORCE(a != nullptr && b != nullptr);

  MatMulComputeHelper helper;
  ORT_RETURN_IF_ERROR(helper.Compute(a->Shape(), b->Shape()));
  Tensor* y = ctx->Output(0, helper.OutputShape());

  // validate zero points
  T1 a_offset = 0;
  T2 b_offset = 0;
  if (has_a_zero_point_) {
    auto a_zero_point = ctx->Input<Tensor>(2);
    ORT_ENFORCE(IsScalarOr1ElementVector(a_zero_point),
                "MatmulInteger : input1 zero point must be a scalar or 1D tensor of size 1");
    a_offset = *a_zero_point->template Data<T1>();
  }
  if (has_b_zero_point_) {
    auto b_zero_point = ctx->Input<Tensor>(3);
    ORT_ENFORCE(IsScalarOr1ElementVector(b_zero_point),
                "MatmulInteger : input2 zero point must be a scalar or 1D tensor of size 1");
    b_offset = *b_zero_point->template Data<T2>();
  }

  auto* multiplier = ctx->Input<Tensor>(4);

  for (size_t i = 0; i < helper.OutputOffsets().size(); i++) {

#if 1
    MLAS_GEMM_U8X8_PARAMETERS gemm_parameters = { };

//    float dequant_scale = 1.0f;

    gemm_parameters.M = static_cast<int>(helper.M());
    gemm_parameters.N = static_cast<int>(helper.N());
    gemm_parameters.K = static_cast<int>(helper.K());
    gemm_parameters.A = a->template Data<T1>() + helper.LeftOffsets()[i];
    gemm_parameters.lda = static_cast<int>(helper.K());
    gemm_parameters.offa = a_offset;
    gemm_parameters.B = (const uint8_t*)b->template Data<T2>() + helper.RightOffsets()[i];
    gemm_parameters.ldb = static_cast<int>(helper.N());
    gemm_parameters.offb = b_offset;
    gemm_parameters.BTypeIsSigned = true;
    gemm_parameters.C = (int32_t*)y->template MutableData<float>() + helper.OutputOffsets()[i];
    gemm_parameters.ldc = static_cast<int>(helper.N());
    gemm_parameters.CTypeIsFloat = true;
    gemm_parameters.Multiplier = multiplier->template Data<float>();
    gemm_parameters.Bias = nullptr;

    MlasGemm(&gemm_parameters, thread_pool);
#else

    QGemm(static_cast<int>(helper.M()),
          static_cast<int>(helper.N()),
          static_cast<int>(helper.K()),
          a->template Data<T1>() + helper.LeftOffsets()[i],
          static_cast<int>(helper.K()),
          a_offset,
          b->template Data<T2>() + helper.RightOffsets()[i],
          static_cast<int>(helper.N()),
          b_offset,
          y->template MutableData<int32_t>() + helper.OutputOffsets()[i],
          static_cast<int>(helper.N()),
          thread_pool);
#endif
  }
  return Status::OK();
}

ONNX_OPERATOR_TYPED_KERNEL_EX(MegaFusion, kMSDomain, 1, float, kCpuExecutionProvider,
                              KernelDefBuilder().TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()).TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>()),
                              MegaFusion<uint8_t, int8_t>);

}  // namespace contrib
}  // namespace onnxruntime
