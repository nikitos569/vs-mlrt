from vsmlrt import RIFE, RIFEModel, Backend

# src = core.std.BlankClip(format=vs.RGBS)
src = video_in

# backend could be:
#  - CPU Backend.OV_CPU(): the recommended CPU backend; generally faster than ORT-CPU.
#  - CPU Backend.ORT_CPU(num_streams=1, verbosity=2): vs-ort cpu backend.
#  - GPU Backend.ORT_CUDA(device_id=0, cudnn_benchmark=True, num_streams=1, verbosity=2)
#     - use device_id to select device
#     - set cudnn_benchmark=False to reduce script reload latency when debugging, but with slight throughput performance penalty.
#  - GPU Backend.TRT(fp16=True, device_id=0, num_streams=1): TensorRT runtime, the fastest NV GPU runtime.
flt = RIFE(src, model=RIFEModel.v4_4, backend=Backend.ORT_CUDA())

flt.set_output()
