#include "ort_test_session.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <assert.h>
#include "providers.h"
#include "TestCase.h"

#ifdef _WIN32
#define strdup _strdup
#endif
extern const OrtApi* c_api;

namespace onnxruntime {
namespace perftest {

const std::string &SampleLoader::Name() const {
  return test_case_->GetTestCaseName();
}
size_t SampleLoader::TotalSampleCount() {
  return test_case_->GetDataCount();
}
size_t SampleLoader::PerformanceSampleCount() {
  return test_case_->GetDataCount();
}
void SampleLoader::LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex> &samples) {
  std::unordered_map<std::string, OrtValue*> feeds;
  for(const mlperf::QuerySampleIndex& test_data_id:samples) {
    test_case_->LoadTestData(test_data_id /* id */, b_, feeds, true);
    OrtValue **input_list = inputs_.data() + test_data_id * input_length_;
    // Discard the names in feeds
    for (size_t i = 0; i != input_length_; ++i) {
      auto iter = feeds.find(input_names_[i]);
      if (iter == feeds.end()) {
        std::ostringstream oss;
        oss << "there is no test input data for input " << input_names_[i] << " and model "
            << test_case_->GetTestCaseName() << std::endl;
        throw std::runtime_error(oss.str());
      }
      input_list[i] = iter->second;
    }
  }
}

void SampleLoader::UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex> &samples) {
  for(const mlperf::QuerySampleIndex& test_data_id:samples) {
    OrtValue **input_list = inputs_.data() + test_data_id * input_length_;
    for (size_t i = 0; i != input_length_; ++i) {
      c_api->ReleaseValue(input_list[i]);
    }
  }
}

SampleLoader::SampleLoader(OrtSession* sess, ITestCase* test_case): test_case_(test_case){
  Ort::ThrowOnError(c_api->SessionGetInputCount(sess, &input_length_));
  OrtAllocator* alloc;
  Ort::ThrowOnError(c_api->GetAllocatorWithDefaultOptions(&alloc));
  input_names_.resize(input_length_);
  for (size_t i = 0; i != input_length_; ++i) {
    char* input_name;
    Ort::ThrowOnError(c_api->SessionGetInputName(sess, i, alloc,&input_name));
    assert(input_name != nullptr);
    input_names_[i] = input_name;
    alloc->Free(alloc, input_name);
  }

  inputs_.resize(test_case_->GetDataCount() * input_length_);
}

OnnxRuntimeTestSession::OnnxRuntimeTestSession(OrtSession* sess, SampleLoader* sample_loader, std::random_device& rd):
sess_(sess), sample_loader_(sample_loader),rand_engine_(rd()){
  Ort::ThrowOnError(c_api->SessionGetInputCount(sess, &input_length_));
  OrtAllocator* alloc;
  Ort::ThrowOnError(c_api->GetAllocatorWithDefaultOptions(&alloc));
  input_names_.resize(input_length_);
  for (size_t i = 0; i != input_length_; ++i) {
    char* input_name;
    Ort::ThrowOnError(c_api->SessionGetInputName(sess, i, alloc,&input_name));
    assert(input_name != nullptr);
    input_names_[i] = input_name;
    alloc->Free(alloc, input_name);
  }

  size_t output_count;
  Ort::ThrowOnError(c_api->SessionGetOutputCount(sess, &output_count));
  output_names_.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    char* output_name;
    Ort::ThrowOnError(c_api->SessionGetOutputName(sess, i, alloc,&output_name));
    assert(output_name != nullptr);
    output_names_[i] = output_name;
    alloc->Free(alloc, output_name);
  }
  output_names_raw_ptr.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    output_names_raw_ptr[i] = output_names_[i].c_str();
  }

}

void OnnxRuntimeTestSession::IssueQuery(const std::vector<mlperf::QuerySample>& samples)  {
  mlperf::QuerySampleResponse res;
  size_t output_count = output_names_.size();
  OrtValue* outputs[output_count];
  memset(outputs,0,output_count * sizeof(OrtValue*));
  for(const mlperf::QuerySample& s: samples){
    Ort::ThrowOnError(c_api->Run(sess_,nullptr,input_names_.data(),sample_loader_->GetInput(s.index),
        input_names_.size(),output_names_raw_ptr.data(), output_names_raw_ptr.size(),outputs));
    for(size_t i=0;i!=output_names_.size(); ++i){
      c_api->ReleaseValue(outputs[i]);
    }
    res.id = s.id;
    res.data = 0;
    res.size = 0;
    mlperf::QuerySamplesComplete(&res,1);
  }
}

bool OnnxRuntimeTestSession::PopulateGeneratedInputTestData()
{
#if 0
  // iterate over all input nodes
  for (size_t i = 0; i < static_cast<size_t>(input_length_); i++) {
    Ort::TypeInfo type_info = session_.GetInputTypeInfo(i);
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    if (type_info.GetONNXType() == ONNX_TYPE_TENSOR) {
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> input_node_dim = tensor_info.GetShape();

        // free dimensions are treated as 1
        for (int64_t& dim : input_node_dim) {
          if (dim == -1) {
            dim = 1;
          }
        }
        // default allocator doesn't have to be freed by user
        auto allocator = static_cast<OrtAllocator*>(Ort::AllocatorWithDefaultOptions());
        Ort::Value input_tensor = Ort::Value::CreateTensor(allocator, (const int64_t*)input_node_dim.data(), input_node_dim.size(), tensor_info.GetElementType());
        PreLoadTestData(0, i, input_tensor.release());
    }
  }
#endif
  return true;
}

OrtSession* CreateOrtSession(Ort::Env& env,
                      const PerformanceTestConfig& performance_test_config){
  OrtSessionOptions* session_options;
  Ort::ThrowOnError(c_api->CreateSessionOptions(&session_options));
  const std::string& provider_name = performance_test_config.machine_config.provider_type_name;
  if (provider_name == onnxruntime::kDnnlExecutionProvider) {
#ifdef USE_DNNL
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Dnnl(session_options, performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("DNNL is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNGraphExecutionProvider) {
#ifdef USE_NGRAPH
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_NGraph(session_options, "CPU"));
#else
    ORT_THROW("nGraph is not supported in this build");
#endif
  } else if (provider_name == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#else
    ORT_THROW("CUDA is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNupharExecutionProvider) {
#ifdef USE_NUPHAR
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nuphar(session_options, /*allow_unaligned_buffers*/ 1, ""));
#else
    ORT_THROW("Nuphar is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0));
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0));
#else
    ORT_THROW("TensorRT is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kOpenVINOExecutionProvider) {
#ifdef USE_OPENVINO
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_OpenVINO(session_options, ""));
#else
    ORT_THROW("OpenVINO is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kNnapiExecutionProvider) {
#ifdef USE_NNAPI
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options));
#else
    ORT_THROW("NNAPI is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kDmlExecutionProvider) {
#ifdef USE_DML
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(session_options, 0));
#else
    ORT_THROW("DirectML is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kAclExecutionProvider) {
#ifdef USE_ACL
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_ACL(session_options,
                                                                   performance_test_config.run_config.enable_cpu_mem_arena ? 1 : 0));
#else
    ORT_THROW("Acl is not supported in this build\n");
#endif
  } else if (provider_name == onnxruntime::kMIGraphXExecutionProvider) {
#ifdef USE_MIGRAPHX
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_MIGraphX(session_options, 0));
#else
    ORT_THROW("MIGraphX is not supported in this build\n");
#endif
  } else if (!provider_name.empty() && provider_name != onnxruntime::kCpuExecutionProvider) {
    ORT_THROW("This backend is not included in perf test runner.\n");
  }

  if (performance_test_config.run_config.enable_cpu_mem_arena)
    Ort::ThrowOnError(c_api->EnableCpuMemArena(session_options));
  else
    Ort::ThrowOnError(c_api->DisableCpuMemArena(session_options));

  if (performance_test_config.run_config.enable_memory_pattern &&
      performance_test_config.run_config.execution_mode == ExecutionMode::ORT_SEQUENTIAL)
    Ort::ThrowOnError(c_api->EnableMemPattern(session_options));
  else
    Ort::ThrowOnError(c_api->DisableMemPattern(session_options));

  Ort::ThrowOnError(c_api->SetSessionExecutionMode(session_options, performance_test_config.run_config.execution_mode));

  if(performance_test_config.run_config.intra_op_num_threads > 0){
    fprintf(stdout, "Setting intra_op_num_threads to %d\n",   performance_test_config.run_config.intra_op_num_threads);
    //TODO: If ORT depends on openmp, we should call omp_set_num_threads instead
    Ort::ThrowOnError(c_api->SetIntraOpNumThreads(session_options, performance_test_config.run_config.intra_op_num_threads));
  }

  if (performance_test_config.run_config.execution_mode == ExecutionMode::ORT_PARALLEL && performance_test_config.run_config.inter_op_num_threads > 0) {
    fprintf(stdout, "Setting inter_op_num_threads to %d\n", performance_test_config.run_config.inter_op_num_threads);
    Ort::ThrowOnError(c_api->SetInterOpNumThreads(session_options, performance_test_config.run_config.inter_op_num_threads));
  }

  // Set optimization level.
  Ort::ThrowOnError(c_api->SetSessionGraphOptimizationLevel(session_options, performance_test_config.run_config.optimization_level));
  if (!performance_test_config.run_config.profile_file.empty())
    Ort::ThrowOnError(c_api->EnableProfiling(session_options,performance_test_config.run_config.profile_file.c_str()));
  if (!performance_test_config.run_config.optimized_model_path.empty())
    Ort::ThrowOnError(c_api->SetOptimizedModelFilePath(session_options, performance_test_config.run_config.optimized_model_path.c_str()));
  OrtSession* ret;
  Ort::ThrowOnError(c_api->CreateSession(env, performance_test_config.model_info.model_file_path.c_str(), session_options, &ret));
  c_api->ReleaseSessionOptions(session_options);
  return ret;
}

}  // namespace perftest
}  // namespace onnxruntime
