# ThermalDepthFusion

Overlay the **Arducam ToF depth** feed onto a **static thermal image** feed and
capture a pixel-aligned thermal + depth bundle.

- **Base (static):** thermal camera (InfiRay P2, 256×192) as a plain **image**
  feed (YUY2 — *not* temperature mode).
- **Movable overlay:** Arducam ToF depth (CSI, 240×180 float mm) + confidence.
- **One window, three views:** raw thermal feed, raw depth feed (both separate),
  and the fused depth-on-thermal overlay used for calibration.

## Per-point calibration model

There is **no whole-frame** min/max transform. You align the movable depth
overlay onto the thermal image at a **near** and a **far** distance; then at run
time **each depth pixel is placed by interpolating between its own MIN-aligned
and MAX-aligned position using that pixel's own depth** — so every point travels
along its individual direction (the `A_i + B_i·f(d)` idea from the VL53 app).

### Workflow (use your hand as the target)

1. **Cal** (or `c`) → starts on the **MIN** slot.
2. Hold your hand at a **near** distance. Use **arrows** (move), **+/−**
   (scale), **[ ]** (rotate), and **h/v** (flip depth if mirrored) until the
   depth blob sits on the thermal hand. Press **n** to record the distance.
3. **m** → **MAX** slot. Move your hand **far**, align again, **n**.
4. **s** to save (`thermal_depth_calibration.json`, auto-loaded next run).
5. **Cal**/`c` to exit — now each depth point is placed by its own depth.

## Build & run

```bash
make
make run                 # needs thermal (USB) + Arducam ToF (CSI, 5V/2A)
```

If the app hangs on the banner (wedged thermal cam after an unclean exit):

```bash
./reset_thermal.sh
```

**Exit cleanly** with Quit / `q` / Ctrl+C — never Ctrl+Z or `kill -9` (that
wedges the USB thermal camera).

## Controls

| | |
|---|---|
| `c` Cal · `m` MIN/MAX slot · `n` set slot dist | `h`/`v` flip depth · `p` palette |
| arrows move · `+`/`-` scale · `[` `]` rotate | `,`/`.` overlay opacity |
| `s` save · `l` load · `r` reset slot | `space` capture · `q`/Esc quit |

## Capture (`space`)

Per shot `fusion_<timestamp>_…`: `_thermal.png` (static base), `_overlay.png`
(fused), `_depth_raw.png` (raw depth feed), `_depth_f32_256x192.raw` (per-point
aligned depth in **mm**, float32, on the thermal grid), `_meta.json`.
