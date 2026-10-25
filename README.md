# PNG to Gerber - PCB Art

An interactive desktop app that converts PNG artwork into KiCad-compatible
Gerber layers for PCB art: exposed copper, solder mask openings, silkscreen,
and a smooth board outline (Edge.Cuts).

## How it works

1. **Open images**. `File > Add image` (Ctrl+I) adds a base image; repeat it
   for as many as you need. `File > Open image` (Ctrl+O) replaces the image
   that is selected in the Layers pane and keeps that image's layers.
   Transparency is preserved; fully transparent pixels never match a layer
   color. Images of different sizes share one canvas, aligned at the top-left.
   The canvas is as wide as the widest image and as tall as the tallest, and
   the board size in millimeters maps onto that whole canvas.
2. **Add layers** with the *Add layer* button in the Layers pane. Each layer
   masks a single color of the base image and is one of:
   - **Copper** - exposed copper artwork
   - **Solder mask opening** - areas where the mask is removed
   - **Silkscreen** - printed legend artwork
   - **Cut** - one closed board outline. A project may contain any number of
     cut layers. Each one must be a single contiguous perimeter; export
     unions them into the Edge.Cuts file.
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
5. **Manage layers**: each base image is a row in the layer list, and the
   layers generated from it are nested underneath (indented). With one image
   the row is titled **Base image**; with several, each row shows its file
   name. Click a base image to add another image, replace that image, remove
   it (and its layers), remove the background, set background tolerance,
   or flatten its colors.
   A base row cannot be moved or duplicated. Click a layer to edit its mask.
   New layers are added under the image that owns the current row. Each row
   has an **eye** on the left: click the eye to show or hide that layer (or
   that base image) without changing the selection. The canvas draws every
   visible base image, top-left aligned, with later images painted over
   earlier ones. Rename a layer by double-clicking its name; reorder with
   Up/Down within its image; delete with Delete; **Duplicate** copies the
   selected layer, including a negative's sources. The **Type** menu on a
   layer switches it between copper, solder mask opening, silkscreen, and
   cut. **Invert** paints the opaque pixels that layer does not cover.
   **Checker opacity** on the toolbar fades the transparency checkerboard;
   it starts at 50%.
6. **Generate a negative layer**: Ctrl/Shift-click several layers in the
   Layers pane (they may belong to different images), then *Generate negative*
   and choose copper, mask opening, or silkscreen. The new layer is nested
   under the image that owns the current row. It covers the opaque pixels of
   the source images that the selected layers do not, and it updates live
   when a source layer changes. Deleting a source removes it from the
   negative; a negative left with no sources is deleted too.
7. **Export** from `File > Export`. **Gerber files** (Ctrl+E) writes one file
   per layer into the output folder. **Gerber ZIP** (Ctrl+Shift+E) writes
   those same files into `<name>.zip`, named with the project name (the saved
   project file name, or the base name in the dialog). Choose the output
   folder, base name, and physical board size in millimeters (aspect ratio can
   be locked to the shared canvas). Visible layers with the same type and
   side are merged into one standard file, taking the union across every
   base image:

   | Layer | File |
   |---|---|
   | Front copper | `<name>-F_Cu.gtl` |
   | Back copper | `<name>-B_Cu.gbl` |
   | Front mask openings | `<name>-F_Mask.gts` |
   | Back mask openings | `<name>-B_Mask.gbs` |
   | Front silkscreen | `<name>-F_Silkscreen.gto` |
   | Back silkscreen | `<name>-B_Silkscreen.gbo` |
   | Board outline | `<name>-Edge_Cuts.gm1` |

   **Export image** opens a dialog listing every layer. Check the layers to
   save as their own PNG (solid preview color where the layer is on,
   transparent elsewhere). A cut layer is saved as its perimeter only.
   **Export a fully rendered PCB image** writes one
   extra picture of the finished board: green solder mask, copper showing
   through the mask openings, white silkscreen on top, and a transparent
   margin outside the cut outline. That picture includes every layer that
   has a mask, including layers left unchecked and layers hidden in the view.

## Remove background

Select a base image in the Layers pane and tick *Remove background*. That
makes that image's outer contiguous background color transparent so it can't
match any layer, then crops the image to the remaining artwork. The shared
canvas (and a locked board aspect) follows the cropped size. The background
is the most common border color, grown inward across 4-connected pixels
within the tolerance spin. Same-colored regions enclosed by artwork (the
inside of a letter, a hole in a shape) are kept. Untick to restore the
original image; layers recompute either way. Each image remembers its own
background removal.

## Flatten colors

Select a base image in the Layers pane and tick *Flatten colors*. The image
is reduced to a chosen number of colors (2–12) by brightness, after background
removal, so transparent pixels stay transparent. A bar shows that brightness
range: the histogram underneath is where the image's pixels sit, the colored
segments are the buckets, and the notches are the cutoffs. Drag a notch to
move a boundary; it cannot cross its neighbors. *Auto-place cutoffs* puts the
notches between the strongest brightness peaks. *Reset to even spacing*
spreads them evenly and keeps any colors you picked.

Each bucket has a swatch. It starts as the average of the pixels in that
bucket (an empty bucket is a gray at the middle of its range). Click a swatch
to choose a different color; right-click it to go back to the average.
*Edge smooth* (0–5) cleans speckles and jagged anti-aliased edges without
inventing new colors. *Compare original* shows the image from before
flattening so you can judge the cutoffs; layer masks keep using the flattened
image. Untick *Flatten colors* to restore the source colors. Each image
remembers its own settings.

The top toolbar holds the **Checker opacity** slider (how strong the transparency
grid is; it starts at 50%). Open, save, and export live in the File menu.
Show or hide the base image from the View menu.

## Paint

The left toolbar paints pixels by hand. Select a layer to edit that layer's
mask, or select the base image to edit its pixels. The tools are a
**paintbrush**, a **straight line**, a **rectangle** (click one corner, then
the opposite corner; only the outline is drawn), an **eraser**, and a **fill**
bucket. One **Brush Size** slider, or a typed number, sets the diameter for every tool except fill. Click the active tool
again to pan with the left button; middle-drag pans while a tool is active.

On a layer, paint forces those pixels on and erase forces them off. The
strokes stay when you change tolerance, gap closing, or invert, and they are
enough on their own — a layer does not need a matched color before you can
paint it. Fill clicks an empty area and fills the connected empty region.
On the base image, the color button chooses the painted color (including
transparency). Erase makes pixels transparent. Fill replaces the connected
pixels of the exact same color. Painted base pixels are applied after
background removal and color flattening, so the color you pick is the color
the layers see.

## Project files (.prj)

*File > Save project* (Ctrl+S / Ctrl+Shift+S) writes a human-readable JSON
`.prj` file; *File > Open project* (Ctrl+Shift+O) restores where you left off:

- every layer, nested under its base image: name, type, side, color,
  tolerance, gap closing, min island, fill holes, cut smoothness/perimeter,
  visibility, invert, negative sources, and hand-painted mask pixels
- per image, any pixels painted onto the base image (coordinates are in the
  source image, so they survive background removal and flattening)
- layer order, board width/height, aspect lock, output folder, and base name
  (set via *File > Board settings* or the export dialog)
- per image: background-removal state and tolerance, color flattening
  (on/off, bucket count, cutoffs, color overrides, edge smooth), visibility,
  and the source image path (absolute plus relative to the `.prj`, so a
  project folder can be moved; if an image is missing you are asked to locate it)

A project with one image is saved in the original version-1 layout. A project
with several images is version 2. Version-1 files still open.

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
