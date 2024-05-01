__version__ = "3.20.12"

__all__ = [
    "Backend", "BackendV2",
    "RIFE", "RIFEModel", "RIFEMerge",
    "inference"
]

import copy
from dataclasses import dataclass, field
import enum
from fractions import Fraction
import math
import os
import os.path
import platform
import subprocess
import sys
import tempfile
import time
import typing
import zlib

import vapoursynth as vs
from vapoursynth import core


def get_plugins_path() -> str:
    path = b""

    path = core.migx.Version()["path"]

    assert path != b""

    return os.path.dirname(path).decode()

plugins_path: str = get_plugins_path()
migraphx_driver_path: str = os.path.join(plugins_path, "vsmlrt-hip", "migraphx-driver")
models_path: str = os.path.join(plugins_path, "models")


class Backend:
    @dataclass(frozen=False)
    class MIGX:
        """ backend for amd gpus

        basic performance tuning:
        set fp16 = True
        """

        device_id: int = 0
        fp16: bool = False
        opt_shapes: typing.Optional[typing.Tuple[int, int]] = None
        fast_math: bool = True
        exhaustive_tune: bool = False

        short_path: typing.Optional[bool] = None # True on Windows by default, False otherwise
        custom_env: typing.Dict[str, str] = field(default_factory=lambda: {})
        custom_args: typing.List[str] = field(default_factory=lambda: [])

        # internal backend attributes
        supports_onnx_serialization: bool = False

backendT = typing.Union[
    Backend.MIGX,
]

fallback_backend: typing.Optional[backendT] = None

def get_rife_input(clip: vs.VideoNode) -> typing.List[vs.VideoNode]:
    assert clip.format.sample_type == vs.FLOAT
    gray_format = vs.GRAYS if clip.format.bits_per_sample == 32 else vs.GRAYH


    if (hasattr(core, 'akarin') and
        b"width" in core.akarin.Version()["expr_features"] and
        b"height" in core.akarin.Version()["expr_features"]
    ):
        if b"fp16" in core.akarin.Version()["expr_features"]:
            empty = clip.std.BlankClip(format=gray_format, length=1)
        else:
            empty = clip.std.BlankClip(format=vs.GRAYS, length=1)

        horizontal = bits_as(core.akarin.Expr(empty, 'X 2 * width 1 - / 1 -'), clip)
        vertical = bits_as(core.akarin.Expr(empty, 'Y 2 * height 1 - / 1 -'), clip)
    else:
        empty = clip.std.BlankClip(format=vs.GRAYS, length=1)

        from functools import partial

        def meshgrid_core(n: int, f: vs.VideoFrame, horizontal: bool) -> vs.VideoFrame:
            fout = f.copy()

            is_api4 = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4
            if is_api4:
                mem_view = fout[0]
            else:
                mem_view = fout.get_write_array(0)

            height, width = mem_view.shape

            if horizontal:
                for i in range(height):
                    for j in range(width):
                        mem_view[i, j] = 2 * j / (width - 1) - 1
            else:
                for i in range(height):
                    for j in range(width):
                        mem_view[i, j] = 2 * i / (height - 1) - 1

            return fout

        horizontal = bits_as(core.std.ModifyFrame(empty, empty, partial(meshgrid_core, horizontal=True)), clip)
        vertical = bits_as(core.std.ModifyFrame(empty, empty, partial(meshgrid_core, horizontal=False)), clip)

    horizontal = horizontal.std.Loop(clip.num_frames)
    vertical = vertical.std.Loop(clip.num_frames)

    multiplier_h = clip.std.BlankClip(format=gray_format, color=2/(clip.width-1), keep=True)

    multiplier_w = clip.std.BlankClip(format=gray_format, color=2/(clip.height-1), keep=True)

    return [horizontal, vertical, multiplier_h, multiplier_w]


@enum.unique
class RIFEModel(enum.IntEnum):
    """
    Starting from RIFE v4.12 lite, this interface does not provide forward compatiblity in enum values.
    """

    v4_0 = 40
    v4_2 = 42
    v4_3 = 43
    v4_4 = 44
    v4_5 = 45
    v4_6 = 46
    v4_7 = 47
    v4_8 = 48
    v4_9 = 49
    v4_10 = 410
    v4_11 = 411
    v4_12 = 412
    v4_12_lite = 4121
    v4_13 = 413
    v4_13_lite = 4131
    v4_14 = 414
    v4_14_lite = 4141
    v4_15 = 415
    v4_15_lite = 4151
    v4_16_lite = 4161


def RIFEMerge(
    clipa: vs.VideoNode,
    clipb: vs.VideoNode,
    mask: vs.VideoNode,
    scale: float = 1.0,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: RIFEModel = RIFEModel.v4_4,
    backend: backendT = Backend.OV_CPU(),
    ensemble: bool = False,
    _implementation: typing.Optional[typing.Literal[1, 2]] = None
) -> vs.VideoNode:
    """ temporal MaskedMerge-like interface for the RIFE model

    Its semantics is similar to core.std.MaskedMerge(clipa, clipb, mask, first_plane=True),
    except that it merges the two clips in the time domain and you specify the "mask" based
    on the time point of the resulting clip (range (0,1)) between the two clips.
    """

    func_name = "vsmlrt.RIFEMerge"

    for clip in (clipa, clipb, mask):
        if not isinstance(clip, vs.VideoNode):
            raise TypeError(f'{func_name}: clip must be a clip!')

        if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
            raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    for clip in (clipa, clipb):
        if clip.format.color_family != vs.RGB:
            raise ValueError(f'{func_name}: "clipa" / "clipb" must be of RGB color family')

        if clip.width != mask.width or clip.height != mask.height:
            raise ValueError(f'{func_name}: video dimensions mismatch')

        if clip.num_frames != mask.num_frames:
            raise ValueError(f'{func_name}: number of frames mismatch')

    if mask.format.color_family != vs.GRAY:
        raise ValueError(f'{func_name}: "mask" must be of GRAY color family')

    if tiles is not None or tilesize is not None or overlap is not None:
        raise ValueError(f'{func_name}: tiling is not supported')

    if overlap is None:
        overlap_w = overlap_h = 0
    elif isinstance(overlap, int):
        overlap_w = overlap_h = overlap
    else:
        overlap_w, overlap_h = overlap

    multiple_frac = 32 / Fraction(scale)
    if multiple_frac.denominator != 1:
        raise ValueError(f'{func_name}: (32 / Fraction(scale)) must be an integer')
    multiple = int(multiple_frac.numerator)
    scale = float(Fraction(scale))

    model_major = int(str(int(model))[0])
    model_minor = int(str(int(model))[1:3])
    lite = "_lite" if len(str(int(model))) >= 4 else ""
    version = f"v{model_major}.{model_minor}{lite}{'_ensemble' if ensemble else ''}"

    if (model_major, model_minor) >= (4, 7) and scale != 1.0:
        raise ValueError("not supported")

    network_path = os.path.join(
        models_path,
        "rife_v2",
        f"rife_{version}.onnx"
    )
    if _implementation == 2 and os.path.exists(network_path) and scale == 1.0:
        implementation_version = 2
        multiple = 1 # v2 implements internal padding
        clips = [clipa, clipb, mask]
    else:
        implementation_version = 1

        network_path = os.path.join(
            models_path,
            "rife",
            f"rife_{version}.onnx"
        )

        clips = [clipa, clipb, mask, *get_rife_input(clipa)]

    (tile_w, tile_h), (overlap_w, overlap_h) = calc_tilesize(
        tiles=tiles, tilesize=tilesize,
        width=clip.width, height=clip.height,
        multiple=multiple,
        overlap_w=overlap_w, overlap_h=overlap_h
    )

    if tile_w % multiple != 0 or tile_h % multiple != 0:
        raise ValueError(
            f'{func_name}: tile size must be divisible by {multiple} ({tile_w}, {tile_h})'
        )

    backend = init_backend(
        backend=backend,
        trt_opt_shapes=(tile_w, tile_h)
    )

    if implementation_version == 2:
        if isinstance(backend, Backend.TRT):
            # https://github.com/AmusementClub/vs-mlrt/issues/66#issuecomment-1791986979
            if (4, 0) <= (model_major, model_minor):
                if backend.force_fp16:
                    backend.force_fp16 = False
                    backend.fp16 = True

                backend.custom_args.extend([
                    "--precisionConstraints=obey",
                    "--layerPrecisions=" + (
                        "/Cast_2:fp32,/Cast_3:fp32,/Cast_5:fp32,/Cast_7:fp32,"
                        "/Reciprocal:fp32,/Reciprocal_1:fp32,"
                        "/Mul:fp32,/Mul_1:fp32,/Mul_8:fp32,/Mul_10:fp32,"
                        "/Sub_5:fp32,/Sub_6:fp32,"
                        # generated by TensorRT's onnx parser
                        "ONNXTRT_Broadcast_236:fp32,ONNXTRT_Broadcast_238:fp32,"
                        "ONNXTRT_Broadcast_273:fp32,ONNXTRT_Broadcast_275:fp32,"
                        # TensorRT 9.0 or later
                        "ONNXTRT_Broadcast_*:fp32"
                    )
                ])

    if scale == 1.0:
        return inference_with_fallback(
            clips=clips, network_path=network_path,
            overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
            backend=backend
        )
    elif ensemble or implementation_version != 1:
        raise ValueError(f'{func_name}: currently not supported')
    else:
        import onnx
        from onnx.numpy_helper import from_array, to_array

        onnx_model = onnx.load(network_path)

        resize_counter = 0
        for i in range(len(onnx_model.graph.node)):
            node = onnx_model.graph.node[i]
            if len(node.output) == 1 and node.op_type == "Constant" and node.output[0].startswith("onnx::Resize"):
                resize_counter += 1

                array = to_array(node.attribute[0].t).copy()
                if resize_counter % 3 == 2:
                    array[2:4] /= scale
                else:
                    array[2:4] *= scale
                onnx_model.graph.node[i].attribute[0].t.raw_data = from_array(array).raw_data

        if resize_counter != 11:
            raise ValueError("invalid rife model")

        multiplier_counter = 0
        for i in range(len(onnx_model.graph.node)):
            node = onnx_model.graph.node[i]
            if len(node.output) == 1 and node.op_type == "Constant" and node.output[0].startswith("onnx::Mul"):
                multiplier_counter += 1

                array = to_array(node.attribute[0].t).copy()
                if multiplier_counter % 2 == 1:
                    array /= scale
                else:
                    array *= scale
                onnx_model.graph.node[i].attribute[0].t.raw_data = from_array(array).raw_data

        if multiplier_counter != 7:
            raise ValueError("invalid rife model")

        if backend.supports_onnx_serialization:
            return inference_with_fallback(
                clips=clips, network_path=onnx_model.SerializeToString(),
                overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
                backend=backend, path_is_serialization=True
            )
        else:
            network_path = f"{network_path}_scale{scale!r}.onnx"
            onnx.save(onnx_model, network_path)

            return inference_with_fallback(
                clips=clips, network_path=network_path,
                overlap=(overlap_w, overlap_h), tilesize=(tile_w, tile_h),
                backend=backend
            )


def RIFE(
    clip: vs.VideoNode,
    multi: typing.Union[int, Fraction] = 2,
    scale: float = 1.0,
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    overlap: typing.Optional[typing.Union[int, typing.Tuple[int, int]]] = None,
    model: RIFEModel = RIFEModel.v4_4,
    backend: backendT = Backend.OV_CPU(),
    ensemble: bool = False,
    video_player: bool = False,
    _implementation: typing.Optional[typing.Literal[1, 2]] = None
) -> vs.VideoNode:
    """ RIFE: Real-Time Intermediate Flow Estimation for Video Frame Interpolation

    multi, scale is based on vs-rife.

    For the best results, you need to perform scene detection on the input clip
    (e.g. misc.SCDetect, mv.SCDetection) before passing it to RIFE.
    Also note that the quality of result is strongly dependent on high quality
    scene detection and you might need to tweak the scene detection parameters
    and/or filter to achieve the best quality.

    Args:
        multi: Multiple of the frame counts, can be a fractions.Fraction.
            Default: 2.

        scale: Controls the process resolution for optical flow model.
            32 / fractions.Fraction(scale) must be an integer.
            scale=0.5 is recommended for 4K video.

        _implementation: (None, 1 or 2, experimental and maybe removed in the future)
            Switch between different onnx implementation.
            Implmementation will be selected based on internal heuristic if it is None.
    """

    func_name = "vsmlrt.RIFE"

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(f'{func_name}: "clip" must be a clip!')

    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample not in [16, 32]:
        raise ValueError(f"{func_name}: only constant format 16/32 bit float input supported")

    if clip.format.color_family != vs.RGB:
        raise ValueError(f'{func_name}: "clip" must be of RGB color family')

    if not isinstance(multi, (int, Fraction)):
        raise TypeError(f'{func_name}: "multi" must be an integer or a fractions.Fraction!')

    if tiles is not None or tilesize is not None or overlap is not None:
        raise ValueError(f'{func_name}: tiling is not supported')

    gray_format = vs.GRAYS if clip.format.bits_per_sample == 32 else vs.GRAYH

    if int(multi) == multi:
        multi = int(multi)

        if multi < 2:
            raise ValueError(f'{func_name}: RIFE: multi must be at least 2')

        initial = core.std.Interleave([clip] * (multi - 1))

        terminal = clip.std.DuplicateFrames(frames=clip.num_frames - 1).std.Trim(first=1)
        terminal = core.std.Interleave([terminal] * (multi - 1))

        timepoint = core.std.Interleave([
            clip.std.BlankClip(format=gray_format, color=i/multi, length=1)
            for i in range(1, multi)
        ]).std.Loop(clip.num_frames)

        output0 = RIFEMerge(
            clipa=initial, clipb=terminal, mask=timepoint,
            scale=scale, tiles=tiles, tilesize=tilesize, overlap=overlap,
            model=model, backend=backend, ensemble=ensemble,
            _implementation=_implementation
        )

        clip = bits_as(clip, output0)
        initial = core.std.Interleave([clip] * (multi - 1))

        if hasattr(core, 'akarin') and hasattr(core.akarin, 'Select'):
            output = core.akarin.Select([output0, initial], initial, 'x._SceneChangeNext 1 0 ?')
        else:
            def handler(n: int, f: vs.VideoFrame) -> vs.VideoNode:
                if f.props.get('_SceneChangeNext'):
                    return initial
                return output0
            output = core.std.FrameEval(output0, handler, initial)

        if multi == 2:
            res = core.std.Interleave([clip, output])
        else:
            res = core.std.Interleave([
                clip,
                *(output.std.SelectEvery(cycle=multi-1, offsets=i) for i in range(multi - 1))
            ])

        if clip.fps_num != 0 and clip.fps_den != 0:
            return res.std.AssumeFPS(fpsnum = clip.fps_num * multi, fpsden = clip.fps_den)
        else:
            return res
    else:
        if not hasattr(core, 'akarin') or \
            not hasattr(core.akarin, 'PropExpr') or \
            not hasattr(core.akarin, 'PickFrames'):
            raise RuntimeError(
                'fractional multi requires plugin akarin '
                '(https://github.com/AkarinVS/vapoursynth-plugin/releases)'
                ', version v0.96g or later.')

        if clip.fps_num == 0 or clip.fps_den == 0:
            src_fps = Fraction(1)
        else:
            src_fps = clip.fps

        dst_fps = src_fps * multi
        src_frames = clip.num_frames
        dst_frames = min(int(src_frames * multi), 2 ** 31 - 1)

        duration_rel = src_fps / dst_fps
        dst_duration = duration_rel.numerator
        src_duration = duration_rel.denominator

        # https://github.com/AmusementClub/vs-mlrt/issues/59#issuecomment-1842649342
        if video_player:
            temp = core.std.BlankClip(clip, length=dst_frames, keep=True)

            def left_func(n: int) -> vs.VideoNode:
                return clip[dst_duration * n // src_duration]
            left_clip = core.std.FrameEval(temp, left_func)

            def right_func(n: int) -> vs.VideoNode:
                # no out of range access because of function filter_sc
                return clip[dst_duration * n // src_duration + 1]
            right_clip = core.std.FrameEval(temp, right_func)

            temp_gray = core.std.BlankClip(temp, format=gray_format, keep=True)
            def timepoint_func(n: int) -> vs.VideoNode:
                current_time = dst_duration * n
                left_index = current_time // src_duration
                left_time = src_duration * left_index
                tp = (current_time - left_time) / src_duration
                return temp_gray.std.BlankClip(color=tp, keep=True)
            tp_clip = core.std.FrameEval(temp_gray, timepoint_func)

            output0 = RIFEMerge(
                clipa=left_clip, clipb=right_clip, mask=tp_clip,
                scale=scale, tiles=tiles, tilesize=tilesize, overlap=overlap,
                model=model, backend=backend, ensemble=ensemble,
                _implementation=_implementation
            )

            left0 = bits_as(left_clip, output0)

            def filter_sc(n: int, f: vs.VideoFrame) -> vs.VideoNode:
                current_time = dst_duration * n
                left_index = current_time // src_duration
                if (
                    current_time % src_duration == 0 or
                    left_index + 1 >= src_frames or
                    f.props.get("_SceneChangeNext", False)
                ):
                    return left0
                else:
                    return output0

            res = core.std.FrameEval(output0, filter_sc, left0)
        else:
            if not hasattr(core, 'akarin') or \
                not hasattr(core.akarin, 'PropExpr') or \
                not hasattr(core.akarin, 'PickFrames'):
                raise RuntimeError(
                    'fractional multi requires plugin akarin '
                    '(https://github.com/AkarinVS/vapoursynth-plugin/releases)'
                    ', version v0.96g or later.')

            left_indices = []
            right_indices = []
            timepoints = []
            output_indices = []

            for i in range(dst_frames):
                current_time = dst_duration * i
                if current_time % src_duration == 0:
                    output_indices.append(current_time // src_duration)
                else:
                    left_index = current_time // src_duration
                    if left_index + 1 >= src_frames:
                        # approximate last frame with last frame of source
                        output_indices.append(src_frames - 1)
                        break
                    output_indices.append(src_frames + len(timepoints))
                    left_indices.append(left_index)
                    right_indices.append(left_index + 1)
                    left_time = src_duration * left_index
                    tp = (current_time - left_time) / src_duration
                    timepoints.append(tp)

            left_clip = core.akarin.PickFrames(clip, left_indices)
            right_clip = core.akarin.PickFrames(clip, right_indices)
            tp_clip = core.std.BlankClip(clip, format=gray_format, length=len(timepoints))
            tp_clip = tp_clip.akarin.PropExpr(lambda: dict(_tp=timepoints)).akarin.Expr('x._tp')

            output0 = RIFEMerge(
                clipa=left_clip, clipb=right_clip, mask=tp_clip,
                scale=scale, tiles=tiles, tilesize=tilesize, overlap=overlap,
                model=model, backend=backend, ensemble=ensemble,
                _implementation=_implementation
            )

            clip0 = bits_as(clip, output0)
            left0 = bits_as(left_clip, output0)
            output = core.akarin.Select([output0, left0], left0, 'x._SceneChangeNext 1 0 ?')
            res = core.akarin.PickFrames(clip0 + output, output_indices)

        if clip.fps_num != 0 and clip.fps_den != 0:
            return res.std.AssumeFPS(fpsnum = dst_fps.numerator, fpsden = dst_fps.denominator)
        else:
            return res


def get_mxr_path(
    network_path: str,
    opt_shapes: typing.Tuple[int, int],
    fp16: bool,
    fast_math: bool,
    exhaustive_tune: bool,
    device_id: int,
    short_path: typing.Optional[bool]
) -> str:

    with open(network_path, "rb") as file:
        checksum = zlib.adler32(file.read())

    migx_version = core.migx.Version()["migraphx_version_build"].decode()

    try:
        device_name = core.migx.DeviceProperties(device_id)["name"].decode()
        device_name = device_name.replace(' ', '-')
    except AttributeError:
        device_name = f"device{device_id}"

    shape_str = f"{opt_shapes[0]}x{opt_shapes[1]}"

    identity = (
        shape_str +
        ("_fp16" if fp16 else "") +
        ("_fast" if fast_math else "") +
        ("_exhaustive" if exhaustive_tune else "") +
        f"_migx-{migx_version}" +
        f"_{device_name}" +
        f"_{checksum:x}"
    )

    if short_path or (short_path is None and platform.system() == "Windows"):
        dirname, basename = os.path.split(network_path)
        return os.path.join(dirname, f"{zlib.crc32((basename + identity).encode()):x}.mxr")
    else:
        return f"{network_path}.{identity}.mxr"


def migraphx_driver(
    network_path: str,
    channels: int,
    opt_shapes: typing.Tuple[int, int],
    fp16: bool,
    fast_math: bool,
    exhaustive_tune: bool,
    device_id: int,
    input_name: str = "input",
    short_path: typing.Optional[bool] = None,
    custom_env: typing.Dict[str, str] = {},
    custom_args: typing.List[str] = []
) -> str:

    if isinstance(opt_shapes, int):
        opt_shapes = (opt_shapes, opt_shapes)

    mxr_path = get_mxr_path(
        network_path=network_path,
        opt_shapes=opt_shapes,
        fp16=fp16,
        fast_math=fast_math,
        exhaustive_tune=exhaustive_tune,
        device_id=device_id,
        short_path=short_path
    )

    if os.access(mxr_path, mode=os.R_OK):
        return mxr_path

    alter_mxr_path = os.path.join(
        tempfile.gettempdir(),
        os.path.splitdrive(mxr_path)[1][1:]
    )

    if os.access(alter_mxr_path, mode=os.R_OK):
        return alter_mxr_path

    try:
        # test writability
        with open(mxr_path, "w") as f:
            pass
        os.remove(mxr_path)
    except PermissionError:
        print(f"{mxr_path} not writable", file=sys.stderr)
        mxr_path = alter_mxr_path
        dirname = os.path.dirname(mxr_path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print(f"change mxr path to {mxr_path}", file=sys.stderr)

    if device_id != 0:
        raise ValueError('"device_id" must be 0')

    args = [
        migraphx_driver_path,
        "compile",
        "--onnx", f"{network_path}",
        "--gpu",
        # f"--device={device_id}",
        "--optimize",
        "--binary",
        "--output", f"{mxr_path}"
    ]

    args.extend(["--input-dim", f"@{input_name}", "1", f"{channels}", f"{opt_shapes[1]}", f"{opt_shapes[0]}"])

    if fp16:
        args.append("--fp16")

    if not fast_math:
        args.append("--disable-fast-math")

    if exhaustive_tune:
        args.append("--exhaustive-tune")

    args.extend(custom_args)

    subprocess.run(args, env=custom_env, check=True, stdout=sys.stderr)

    return mxr_path


def calc_size(width: int, tiles: int, overlap: int, multiple: int = 1) -> int:
    return math.ceil((width + 2 * overlap * (tiles - 1)) / (tiles * multiple)) * multiple


def calc_tilesize(
    tiles: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    tilesize: typing.Optional[typing.Union[int, typing.Tuple[int, int]]],
    width: int,
    height: int,
    multiple: int,
    overlap_w: int,
    overlap_h: int
) -> typing.Tuple[typing.Tuple[int, int], typing.Tuple[int, int]]:

    if tilesize is None:
        if tiles is None:
            overlap_w = 0
            overlap_h = 0
            tile_w = width
            tile_h = height
        elif isinstance(tiles, int):
            tile_w = calc_size(width, tiles, overlap_w, multiple)
            tile_h = calc_size(height, tiles, overlap_h, multiple)
        else:
            tile_w = calc_size(width, tiles[0], overlap_w, multiple)
            tile_h = calc_size(height, tiles[1], overlap_h, multiple)
    elif isinstance(tilesize, int):
        tile_w = tilesize
        tile_h = tilesize
    else:
        tile_w, tile_h = tilesize

    return (tile_w, tile_h), (overlap_w, overlap_h)


def init_backend(
    backend: backendT,
    trt_opt_shapes: typing.Tuple[int, int]
) -> backendT:

    if backend is Backend.MIGX: # type: ignore
        backend = Backend.MIGX()

    backend = copy.deepcopy(backend)

    if isinstance(backend, Backend.MIGX):
        if backend.opt_shapes is None:
            backend.opt_shapes = trt_opt_shapes

    return backend


def _inference(
    clips: typing.List[vs.VideoNode],
    network_path: typing.Union[bytes, str],
    overlap: typing.Tuple[int, int],
    tilesize: typing.Tuple[int, int],
    backend: backendT,
    path_is_serialization: bool = False,
    input_name: str = "input"
) -> vs.VideoNode:

    if not path_is_serialization:
        network_path = typing.cast(str, network_path)
        if not os.path.exists(network_path):
            raise RuntimeError(
                f'"{network_path}" not found, '
                "built-in models can be found at"
                "https://github.com/AmusementClub/vs-mlrt/releases/tag/model-20211209, "
                "https://github.com/AmusementClub/vs-mlrt/releases/tag/model-20220923 and "
                "https://github.com/AmusementClub/vs-mlrt/releases/tag/external-models"
            )

    if isinstance(backend, Backend.MIGX):
        if path_is_serialization:
            raise ValueError('"path_is_serialization" must be False for migx backend')

        network_path = typing.cast(str, network_path)

        channels = sum(clip.format.num_planes for clip in clips)

        opt_shapes = backend.opt_shapes if backend.opt_shapes is not None else tilesize

        mxr_path = migraphx_driver(
            network_path,
            channels=channels,
            opt_shapes=opt_shapes,
            fp16=backend.fp16,
            fast_math=backend.fast_math,
            exhaustive_tune=backend.exhaustive_tune,
            device_id=backend.device_id,
            input_name=input_name,
            short_path=backend.short_path,
            custom_env=backend.custom_env,
            custom_args=backend.custom_args
        )
        clip = core.migx.Model(
            clips, mxr_path,
            overlap=overlap,
            tilesize=tilesize,
            device_id=backend.device_id
        )
    else:
        raise TypeError(f'unknown backend {backend}')

    return clip


def inference_with_fallback(
    clips: typing.List[vs.VideoNode],
    network_path: typing.Union[bytes, str],
    overlap: typing.Tuple[int, int],
    tilesize: typing.Tuple[int, int],
    backend: backendT,
    path_is_serialization: bool = False,
    input_name: str = "input"
) -> vs.VideoNode:

    try:
        return _inference(
            clips=clips, network_path=network_path,
            overlap=overlap, tilesize=tilesize,
            backend=backend,
            path_is_serialization=path_is_serialization,
            input_name=input_name
        )
    except Exception as e:
        if fallback_backend is not None:
            import logging
            logger = logging.getLogger("vsmlrt")
            logger.warning(f'"{backend}" fails, trying fallback backend "{fallback_backend}"')

            return _inference(
                clips=clips, network_path=network_path,
                overlap=overlap, tilesize=tilesize,
                backend=fallback_backend,
                path_is_serialization=path_is_serialization,
                input_name=input_name
            )
        else:
            raise e


def inference(
    clips: typing.Union[vs.VideoNode, typing.List[vs.VideoNode]],
    network_path: str,
    overlap: typing.Tuple[int, int] = (0, 0),
    tilesize: typing.Optional[typing.Tuple[int, int]] = None,
    backend: backendT = Backend.MIGX(),
    input_name: typing.Optional[str] = "input"
) -> vs.VideoNode:

    if isinstance(clips, vs.VideoNode):
        clips = typing.cast(vs.VideoNode, clips)
        clips = [clips]

    if tilesize is None:
        tilesize = (clips[0].width, clips[0].height)

    backend = init_backend(backend=backend, trt_opt_shapes=tilesize)

    if input_name is None:
        input_name = get_input_name(network_path)

    return inference_with_fallback(
        clips=clips,
        network_path=network_path,
        overlap=overlap,
        tilesize=tilesize,
        backend=backend,
        path_is_serialization=False,
        input_name=input_name
    )


def get_input_name(network_path: str) -> str:
    import onnx
    model = onnx.load(network_path)
    return model.graph.input[0].name


def bits_as(clip: vs.VideoNode, target: vs.VideoNode) -> vs.VideoNode:
    if clip.format.bits_per_sample == target.format.bits_per_sample:
        return clip
    else:
        is_api4 = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4
        query_video_format = core.query_video_format if is_api4 else core.register_format
        format = query_video_format(
            color_family=clip.format.color_family,
            sample_type=clip.format.sample_type,
            bits_per_sample=target.format.bits_per_sample,
            subsampling_w=clip.format.subsampling_w,
            subsampling_h=clip.format.subsampling_h
        )
        return clip.resize.Point(format=format)


class BackendV2:
    """ simplified backend interfaces with keyword-only arguments

    More exposed arguments may be added for each backend,
    but existing ones will always function in a forward compatible way.
    """

    @staticmethod
    def MIGX(*,
        fp16: bool = False,
        opt_shapes: typing.Optional[typing.Tuple[int, int]] = None,
        **kwargs
    ) -> Backend.MIGX:

        return Backend.MIGX(
            fp16=fp16,
            opt_shapes=opt_shapes
            **kwargs
        )

def fmtc_resample(clip: vs.VideoNode, **kwargs) -> vs.VideoNode:
    clip_org = clip

    if clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample != 32:
        format = clip.format.replace(core=core, bits_per_sample=32)
        clip = core.resize.Point(clip, format=format.id)

    clip = core.fmtc.resample(clip, **kwargs)

    if clip.format.bits_per_sample != clip_org.format.bits_per_sample:
        clip = core.resize.Point(clip, format=clip_org.format.id)

    return clip
