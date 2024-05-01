from vsmlrt import RIFE, RIFEModel, Backend

# src = core.std.BlankClip(format=vs.RGBS)
src = video_in

flt = RIFE(src, model=RIFEModel.v4_4, backend=Backend.MIGX())

flt.set_output()
