#include "matx.h"
#include <nvbench/nvbench.cuh>

using namespace matx;

using conv_types =
    nvbench::type_list<cuda::std::complex<float>, cuda::std::complex<double>>;

/* FFT benchmarks */
template <typename ValueType>
void conv1d_4d_batch(nvbench::state &state,
                            nvbench::type_list<ValueType>)
{


  auto out = make_tensor<ValueType>({4, 2, 14, 288 + 4096 + 133 - 1});
  auto at = make_tensor<ValueType>({ 4, 2, 14, 133});
  auto bt = make_tensor<ValueType>({ 4, 2, 14, 288 + 4096});

  out.PrefetchDevice(0);
  at.PrefetchDevice(0);
  bt.PrefetchDevice(0);

  state.exec(
      [&out, &at, &bt](nvbench::launch &launch) { conv1d(out, at, bt, MATX_C_MODE_FULL, launch.get_stream()); });
}
NVBENCH_BENCH_TYPES(conv1d_4d_batch, NVBENCH_TYPE_AXES(conv_types));


template <typename ValueType>
void conv1d_2d_batch(nvbench::state &state,
                            nvbench::type_list<ValueType>)
{


  auto out = make_tensor<ValueType>({4 * 2* 14, 288 + 4096 + 133 - 1});
  auto at = make_tensor<ValueType>({ 4 * 2* 14, 133});
  auto bt = make_tensor<ValueType>({ 4 * 2* 14, 288 + 4096});

  out.PrefetchDevice(0);
  at.PrefetchDevice(0);
  bt.PrefetchDevice(0);

  state.exec(
      [&out, &at, &bt](nvbench::launch &launch) { conv1d(out, at, bt, MATX_C_MODE_FULL, launch.get_stream()); });
}
NVBENCH_BENCH_TYPES(conv1d_2d_batch, NVBENCH_TYPE_AXES(conv_types));