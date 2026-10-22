# Desk View

A top-down view of your desk from an ordinary webcam, like macOS Desk View, but calibrated
against a printed chart instead of a guessed camera position.

macOS Desk View assumes a fixed camera geometry, so the result looks skewed whenever the
camera angle, the laptop lid angle or the lens differs from what it expects. This tool
measures all of that:

- **Lens calibration** (optional, recommended): about 20 views of the chart give the focal
  length and lens distortion. Wide-angle and Continuity Camera lenses bend straight lines,
  and this step removes the bend.
- **Desk-plane calibration**: the chart lying flat on the desk gives the exact pose of the
  desk relative to the camera, averaged over 30 frames to cancel sensor noise.
- **One-step warp**: each output pixel maps straight back through the lens model to a camera
  pixel, so undistortion and rectification happen in a single resample. Distances in the
  view are true to scale (mm on the desk), and you can pan, zoom and rotate without
  recalibrating.

It reports how well it fits (RMS error in pixels and in mm on the desk) so you can tell how
good the calibration is.

## Setup

```sh
uv sync                     # or: uv sync --extra virtualcam   (see below)
uv run desk-view
```

On macOS, give your terminal camera access the first time it asks (System Settings →
Privacy & Security → Camera).

## Widest field of view

On macOS the app talks to AVFoundation directly instead of going through OpenCV, so it can:

- pick the camera format that covers the most of the sensor. On a MacBook Pro camera that is
  the 1552×1552 mode, which sees about 47% more of the desk vertically than the default
  1920×1080 stream;
- reset the zoom to its minimum (the built-in camera defaults to 1.5×);
- turn off Center Stage and geometric distortion correction, which both crop. The lens
  calibration corrects distortion instead.

The status bar shows the mode in use.

## Calibrating

1. Print a chart from [`calibration_chart/`](calibration_chart/) at 100% scale.
2. Measure one black square and enter it in **Square size** (36 mm if it printed exactly).
3. Click **Calibrate…** (or **Start calibration** on first launch).
   - **Step 1, lens:** hold the chart and move it around the frame. Go near and far, into
     every corner, and tilt it in each direction. Views are captured automatically when the
     pose is new, and green shading shows which parts of the image are covered. **Skip lens
     step** estimates focal length from the desk view alone, which is less accurate.
     **Reuse previous lens calibration** appears if you've done this before, so you can just
     redo the desk after moving the laptop.
   - **Step 2, desk:** lay the chart flat in the middle of the area you want to show and
     click **Lock in**. Any rotation works, because "up" is aligned with the camera
     automatically.

Calibrations are stored per camera in `~/.config/desk_view/cameras/`.

## Using the view

| Control | Action |
| --- | --- |
| drag / arrow keys | pan (click the view first for keys) |
| scroll / `+` `-` | zoom |
| **Fit** | largest sharp area that is fully in view |
| **Rotation** box | which way is "up": degrees clockwise from the camera-aligned default (0°) |
| Shift+drag / `[` `]` | rotate freely / by 1° (updates the Rotation box) |
| **Rotate 180°** | flip the view the right way up for the other person |
| **Mirror** | mirror the view |
| **Desk / Camera** | switch to the raw camera feed (the desk area is outlined) |

View changes are saved automatically.

### Use in video calls

```sh
uv sync --extra virtualcam
```

Then tick **Virtual camera**. On macOS this needs [OBS](https://obsproject.com) installed,
with its virtual camera started once. Pick "OBS Virtual Camera" in Zoom, Meet or FaceTime.

## Tips

- Desk View works best with a wide camera tilted down toward the desk. An iPhone used as a
  Continuity Camera (ultra-wide) works well. A laptop's built-in camera only sees the part
  of the desk near the screen.
- The desk view is rendered at 1280×720.
