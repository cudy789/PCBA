# PNG to Gerber - PCB Art

An interactive desktop app that converts PNG artwork into KiCad-compatible
Gerber layers for PCB art: exposed copper, solder mask openings, silkscreen,
and a smooth board outline (Edge.Cuts).

## How it works

1. **Open an image** (`File > Open image`, Ctrl+O). Transparency is preserved;
   fully transparent pixels never match a layer color.
2. **Add layers** with the *Add layer* button in the Layers pane. Each layer
   masks a single color of the base image and is one of:
   - **Copper** - exposed copper artwork
   - **Solder mask opening** - areas where the mask is removed
   - **Silkscreen** - printed legend artwork
   - **Cut** - the board outline (only one allowed; it must be a single
     contiguous perimeter)
3. **Pick the layer's color** with any of:
   - the **eyedropper** (click a pixel on the canvas),
   - the **RGBA color dialog** (click the color swatch button),
   - a **hex code** (`#RRGGBB`),
   - the **dominant image color swatches** shown in the inspector.
4. **Tune the mask** and watch the tinted preview update in real time:
   - *Tolerance* - perceptual color distance, so anti-aliased pixels that are
     similar-but-not-identical to the base color are included and the mask
     stays free of pixel-size gaps.
   - *Gap closing* - morphological closing radius that fills pinholes.
   - *Min island px* - removes tiny speckles.
   - *Fill enclosed holes* - fills fully-surrounded holes in the mask.
   - *Cut perimeter* (cut layer only) - **Outer** traces the outside edge of
     the selected color's largest region; **Inner** traces the edge of the
     largest area the color encloses (e.g. the inside of a frame or border
     line drawn around the board). Inner needs a closed ring - if the color
     has gaps, raise tolerance/gap closing.
   - *Cut smoothness* (cut layer only) - a 0-100 slider that smooths jagged
     pixel steps into a clean, rounded perimeter.
5. **Manage layers**: **Base image** is the first row of the layer list.
   Click it to open/replace the image, remove the background, and set
   background tolerance. It stays put: it cannot be moved, duplicated, or
   deleted. Click a layer to edit its mask. Each row has an **eye** on the
   left: click the eye to show or hide that layer (or the base image)
   without changing the selection. Rename a layer by double-clicking its
   name; reorder with Up/Down; delete with Delete; **Duplicate** copies the
   selected layer, including a negative's sources. The **Type** menu on a
   layer switches it between copper, solder mask opening, silkscreen, and
   cut (still only one cut). **Invert** paints the opaque pixels that layer
   does not cover. **Checker opacity** on the toolbar fades the
   transparency checkerboard; it starts at 50%.
6. **Generate a negative layer**: Ctrl/Shift-click several layers in the
   Layers pane, then *Generate negative* and choose copper, mask opening, or
   silkscreen. The new layer covers everything the selected layers do not
   (within the opaque part of the image) and updates live when a source layer
   changes. Deleting a source removes it from the negative; a negative left
   with no sources is deleted too.
7. **Export** from `File > Export`. **Gerber files** (Ctrl+E) writes one file
   per layer into the output folder. **Gerber ZIP** (Ctrl+Shift+E) writes
   those same files into `<name>.zip`, named with the project name (the saved
   project file name, or the base name in the dialog). Choose the output
   folder, base name, and physical board size in millimeters (aspect ratio can
   be locked to the image). Visible layers with the same type and side are
   merged into one standard file:

   | Layer | File |
   |---|---|
   | Front copper | `<name>-F_Cu.gtl` |
   | Back copper | `<name>-B_Cu.gbl` |
   | Front mask openings | `<name>-F_Mask.gts` |
   | Back mask openings | `<name>-B_Mask.gbs` |
   | Front silkscreen | `<name>-F_Silkscreen.gto` |
   | Back silkscreen | `<name>-B_Silkscreen.gbo` |
   | Board outline | `<name>-Edge_Cuts.gm1` |

## Remove background

Select **Base image** in the Layers pane and tick *Remove background*. That
makes the image's outer contiguous background color transparent so it can't
match any layer, then crops the image to the remaining artwork. The board
size is mapped onto that cropped image, so the Gerber files fill the width
and height you set instead of leaving the old transparent margin in the
coordinates. The background is the most common border color, grown inward
across 4-connected pixels within the tolerance spin. Same-colored regions
enclosed by artwork (the inside of a letter, a hole in a shape) are kept.
Untick to restore the original image; layers recompute either way.

The toolbar holds the **Checker opacity** slider (how strong the transparency
grid is; it starts at 50%). Open, save, and export live in the File menu.
Show or hide the base image from the View menu.

## Project files (.prj)

*File > Save project* (Ctrl+S / Ctrl+Shift+S) writes a human-readable JSON
`.prj` file; *File > Open project* (Ctrl+Shift+O) restores where you left off:

- every layer: name, type, side, color, tolerance, gap closing, min island,
  fill holes, cut smoothness/perimeter, visibility, invert, and negative sources
- layer order, board width/height, aspect lock, output folder, and base name
  (set via *File > Board settings* or the export dialog)
- background-removal state and tolerance, and base-image visibility
- the source image path (absolute plus relative to the `.prj`, so a project
  folder can be moved; if the image is missing you are asked to locate it)

The image itself is not embedded. Corrupt or unsupported files are rejected
with a message and never overwrite the work currently open.

If the project has unsaved changes, closing the window, opening another
project, or replacing the image asks whether to Save, Discard, or Cancel.

## Gerber semantics

- Gerber X2 with `.FileFunction` attributes, millimeters, absolute 4.6
  coordinates (`%FSLAX46Y46*%`, `%MOMM*%`).
- Artwork is one solid `G36`/`G37` region per outline. KiCad's Gerber viewer
  fills every contour solid (a `D02` starts a new polygon instead of cutting a
  hole), so each hole is a following region with clear polarity (`%LPC%`).
  An island inside a hole is drawn dark again after that clear, which is why
  enclosed areas survive in GerbView.
- The outline is a closed stroked path (0.10 mm aperture) with the
  `Profile,NP` file function; KiCad 10's Gerber viewer identifies every file
  automatically.
- Output is artwork and a board profile only - not routed, DRC-validated
  electronics. Always review the result in the KiCad Gerber Viewer before
  ordering boards.

## Setup

```powershell
.venv\Scripts\python.exe -m pip install -e .[test]
```

or, without installing the package:

```powershell
.venv\Scripts\python.exe -m pip install numpy opencv-python Pillow shapely PySide6
```

## Run

```powershell
.venv\Scripts\python.exe main.py            # or: python main.py my_art.png
```

## Tests

```powershell
.venv\Scripts\python.exe -m pytest
```

GUI tests run headless via Qt's offscreen platform; no display is needed.

## Portable executable (later)

The app is structured for PyInstaller:

```powershell
.venv\Scripts\python.exe -m pip install pyinstaller
.venv\Scripts\pyinstaller --noconfirm --windowed --name png_to_gerber `
    --collect-submodules PySide6 main.py
```

The result lands in `dist/png_to_gerber/`. Use `--onefile` for a single
self-contained `.exe` (slower startup, simpler distribution).
