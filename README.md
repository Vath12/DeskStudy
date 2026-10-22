# Desk View

!!!THIS APP IS VIBECODED!!!
It's meant to work, not be well written.

A calibrated, top-down view of your desk from a webcam, like macOS Desk View.

## Requirements

- macOS
- A camera that can see your desk: built-in, USB, or a phone used as a webcam
- A printer, for the calibration chart
- [uv](https://docs.astral.sh/uv/). It installs the right Python and all dependencies for you:
  ```sh
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Install

```sh
git clone <this repository's URL> desk_view
cd desk_view
```

There's nothing else to install. The first `./run.sh` downloads everything it needs.

## 1. Print the calibration chart

```sh
./generate_calibration_charts.sh
```

This writes `calibration_chart/desk_view_chart_letter.pdf` (US Letter) and
`calibration_chart/desk_view_chart_a4.pdf` (A4). Print the one for your paper size at
**100% / "Actual size"**, not "Fit to page".

Measure the width of one black square with a ruler. It should be 36 mm. Printers often
scale slightly, so write down what you actually measure.

## 2. Start the app

```sh
./run.sh
```

The first time, macOS asks whether your terminal app may use the camera. Allow it. If you
missed the prompt, enable it in **System Settings → Privacy & Security → Camera** and
start the app again.

## 3. Calibrate

1. Pick your camera from the **Camera** menu.
2. Type the square size you measured into **Square size** (in mm).
3. Click **Calibrate…** and follow the two steps shown in the window:
   - **Lens:** hold the chart flat (taped to cardboard helps) and move it around in front
     of the camera: near, far, into every corner, tilted in each direction. Views are
     captured automatically. You can skip this step, but the view is less accurate without it.
   - **Desk:** lay the chart flat on the desk, in the middle of the area you want to see,
     keep your hands clear and hold still. It locks in by itself.

Calibration is saved per camera. Redo the desk step whenever the camera or laptop moves:
click **Calibrate…**, then **Reuse previous lens calibration**.

## Permissions for the browser buttons

The **Browser**, **ChatGPT**, **Claude**, … buttons open your default browser next to Desk
View. The first time, macOS asks whether your terminal may control the browser. Allow it,
or enable it later in **System Settings → Privacy & Security → Automation**.

Chrome, Edge, Brave, Vivaldi, Chromium and Safari need nothing else. For other browsers
(e.g. Firefox), placing the window also needs **System Settings → Privacy & Security →
Accessibility** for your terminal app.

## Optional: use the desk view in video calls

```sh
uv sync --extra virtualcam
```

Then install [OBS](https://obsproject.com) and start its virtual camera once. After that,
tick **Virtual camera** in Desk View and choose "OBS Virtual Camera" in Zoom, Meet or
FaceTime.

## Where files are kept

- `calibration_chart/`: the printable charts (regenerate any time).
- `scans/`: pictures taken with **Scan**.
- `~/.config/desk_view/`: calibrations and settings. Delete it to start over.
