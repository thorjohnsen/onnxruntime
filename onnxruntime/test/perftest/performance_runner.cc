// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: Remove when removing Eigen
#if defined(_MSC_VER)
#pragma warning(disable : 4267)
#endif

#include "performance_runner.h"
#include <iostream>

#include "TestCase.h"
#include "TFModelInfo.h"
#include "utils.h"
#include "ort_test_session.h"
#ifdef HAVE_TENSORFLOW
#include "tf_test_session.h"
#endif
using onnxruntime::Status;

// TODO: Temporary, while we bring up the threadpool impl...
#include "core/platform/threadpool.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <unsupported/Eigen/CXX11/ThreadPool>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <core/platform/thread_pool_interface.h>
#include <core/platform/EigenNonBlockingThreadPool.h>
extern const OrtApi* c_api;


using DefaultThreadPoolType = onnxruntime::ThreadPoolTempl<onnxruntime::Env>;
static std::unique_ptr<DefaultThreadPoolType> default_pool;
static std::once_flag default_pool_init;
static onnxruntime::ThreadOptions to;

onnxruntime::ThreadPoolInterface* GetDefaultThreadPool(const onnxruntime::Env& env) {
  std::call_once(default_pool_init, [&env] {
    int core_num = env.GetNumCpuCores();
    default_pool = onnxruntime::make_unique<DefaultThreadPoolType>(ORT_TSTR("test_runner"), core_num, false, onnxruntime::Env::Default(),to);
  });
  return default_pool.get();
}

namespace onnxruntime {
namespace perftest {
Status PerformanceRunner::Run() {
  if (!Initialize()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "failed to initialize.");
  }

  // warm up
  RunOneIteration<true>();

  // TODO: start profiling
  // if (!performance_test_config_.run_config.profile_file.empty())
  performance_result_.start_ = std::chrono::high_resolution_clock::now();

  std::unique_ptr<utils::ICPUUsage> p_ICPUUsage = utils::CreateICPUUsage();
  switch (performance_test_config_.run_config.test_mode) {
    case TestMode::kFixDurationMode:
      ORT_RETURN_IF_ERROR(FixDurationTest());
      break;
    case TestMode::KFixRepeatedTimesMode:
      ORT_RETURN_IF_ERROR(RepeatedTimesTest());
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "unknown test mode.");
  }
  performance_result_.end_ = std::chrono::high_resolution_clock::now();

  performance_result_.average_CPU_usage = p_ICPUUsage->GetUsage();
  performance_result_.peak_workingset_size = utils::GetPeakWorkingSetSize();

  std::chrono::duration<double> session_create_duration = session_create_end_ - session_create_start_;
  // TODO: end profiling
  // if (!performance_test_config_.run_config.profile_file.empty()) session_object->EndProfiling();
  std::chrono::duration<double> inference_duration = performance_result_.end_ - performance_result_.start_;

  std::cout << "Session creation time cost:" << session_create_duration.count() << " s" << std::endl
            << "Total inference time cost:" << performance_result_.total_time_cost << " s" << std::endl  // sum of time taken by each request
            << "Total inference requests:" << performance_result_.time_costs.size() << std::endl
            << "Average inference time cost:" << performance_result_.total_time_cost / performance_result_.time_costs.size() * 1000 << " ms" << std::endl
            // Time between start and end of run. Less than Total time cost when running requests in parallel.
            << "Total inference run time:" << inference_duration.count() << " s" << std::endl;
  return Status::OK();
}

Status PerformanceRunner::FixDurationTest() {
  if (performance_test_config_.run_config.concurrent_session_runs <= 1) {
    return RunFixDuration();
  }

  return RunParallelDuration();
}

Status PerformanceRunner::RepeatedTimesTest() {
  if (performance_test_config_.run_config.concurrent_session_runs <= 1) {
    return RunRepeatedTimes();
  }

  return ForkJoinRepeat();
}

Status PerformanceRunner::RunParallelDuration() {
  abort();
}

Status PerformanceRunner::ForkJoinRepeat() {
  const auto& run_config = performance_test_config_.run_config;

  // create a threadpool with one thread per concurrent request
  auto tpool = onnxruntime::make_unique<DefaultThreadPoolType>(ORT_TSTR("test_runner"), run_config.concurrent_session_runs, false, onnxruntime::Env::Default(),to);
  std::atomic<int> counter{0}, requests{0};
  std::mutex m;
  std::condition_variable cv;

  // Fork
  for (size_t i = 0; i != run_config.concurrent_session_runs; ++i) {
    counter++;
    tpool->Schedule([this, &counter, &requests, &m, &cv, &run_config]() {
      while (requests++ < static_cast<int>(run_config.repeated_times)) {
        auto status = RunOneIteration<false>();
        if (!status.IsOK())
          std::cerr << status.ErrorMessage();
      }

      // Simplified version of Eigen::Barrier
      std::lock_guard<std::mutex> lg(m);
      counter--;
      cv.notify_all();
    });
  }

  //Join
  std::unique_lock<std::mutex> lock(m);
  cv.wait(lock, [&counter]() { return counter == 0; });

  return Status::OK();
}

static TestModelInfo* CreateModelInfo(const PerformanceTestConfig& performance_test_config_) {
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("ort")) == 0) {
    return TestModelInfo::LoadOnnxModel(performance_test_config_.model_info.model_file_path.c_str());
  }
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("tf")) == 0) {
    return TFModelInfo::Create(performance_test_config_.model_info.model_file_path.c_str());
  }
  ORT_NOT_IMPLEMENTED(ToMBString(performance_test_config_.backend), " is not supported");
}



static TestSession* CreateSession(Ort::Env& env, std::random_device& rd,
                                  const PerformanceTestConfig& performance_test_config_,
                                  TestModelInfo* test_model_info) {
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("ort")) == 0) {
    abort();
  }
#ifdef HAVE_TENSORFLOW
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("tf")) == 0) {
    return new TensorflowTestSession(rd, performance_test_config_, test_model_info);
  }
#endif
  ORT_NOT_IMPLEMENTED(ToMBString(performance_test_config_.backend), " is not supported");
}

static OrtSession* sess{nullptr};

PerformanceRunner::PerformanceRunner(Ort::Env& env, const PerformanceTestConfig& test_config, std::random_device& rd)
    : performance_test_config_(test_config),
      test_model_info_(CreateModelInfo(test_config)) {
  session_create_start_ = std::chrono::high_resolution_clock::now();
  sess = CreateOrtSession(env,test_config);
  session_create_end_ = std::chrono::high_resolution_clock::now();
}

PerformanceRunner::~PerformanceRunner() = default;



bool PerformanceRunner::Initialize() {
  std::basic_string<PATH_CHAR_TYPE> test_case_dir;
  auto st = GetDirNameFromFilePath(performance_test_config_.model_info.model_file_path, test_case_dir);
  if (!st.IsOK()) {
    printf("input path is not a valid model\n");
    return false;
  }
  std::basic_string<PATH_CHAR_TYPE> model_name = GetLastComponent(test_case_dir);
  // TODO: remove the input and model name's dependency on directory tree
  if (CompareCString(model_name.c_str(), ORT_TSTR("test_")) == 0) {
    model_name = model_name.substr(5);
  }
  std::string narrow_model_name = ToMBString(model_name);
  performance_result_.model_name = narrow_model_name;

  test_case_.reset(CreateOnnxTestCase(narrow_model_name, test_model_info_, 0.0, 0.0));

  if (performance_test_config_.run_config.generate_model_input_binding)
  {
    return static_cast<OnnxRuntimeTestSession*>(session_.get())->PopulateGeneratedInputTestData();
  }

  // TODO: Place input tensor on cpu memory if dnnl provider type to avoid CopyTensor logic in CopyInputAcrossDevices
  size_t test_data_count = test_case_->GetDataCount();
  if (test_data_count == 0) {
    std::cout << "there is no test data for model " << test_case_->GetTestCaseName() << std::endl;
    return false;
  }

  std::random_device rd;
  SampleLoader s(sess,test_case_.get());
  session_.reset(new OnnxRuntimeTestSession(sess, &s, rd));

  mlperf::TestSettings test_settings;
  mlperf::LogSettings log_settings;
  mlperf::StartTest(session_.get(),&s,test_settings,log_settings);
  //test_case_.reset(nullptr);
  test_model_info_ = nullptr;
  return true;
}

}  // namespace perftest

}  // namespace onnxruntime
