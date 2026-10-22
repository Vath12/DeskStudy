# Calibration chart

Print **one** of these at **100% / "Actual size"** (turn off "Fit to page" / "Scale to fit"):

| File | Paper |
| --- | --- |
| `desk_view_chart_letter.pdf` | US Letter, landscape |
| `desk_view_chart_a4.pdf` | A4, landscape |

`preview.png` is a low-resolution preview, not for printing. `board.json` records the layout
(a 7×5 ChArUco board, 36 mm squares, 27 mm ArUco `DICT_4X4_50` markers).

## Measure the print

Printers often scale slightly. Measure the width of one black square with a ruler and type it
into **Square size** in the app before you calibrate. The 100 mm ruler at the bottom of the
page is a quick check. If you change the value after calibrating, the view rescales exactly
and you don't need to recalibrate.

Matte paper works best, because glare on glossy paper can hide markers. For the lens step,
tape the print to a piece of cardboard or a clipboard so it stays flat.

## Regenerating

```sh
uv run desk-view chart                    # rewrites this folder
uv run desk-view chart --square-mm 30     # smaller squares, e.g. for a closer camera
```
