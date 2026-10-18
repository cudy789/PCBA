import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
sys.path.insert(0, "src")
import numpy as np
from PIL import Image
from PySide6.QtWidgets import QApplication
from png_to_gerber.gui.main_window import MainWindow
from png_to_gerber.model import LayerType

app = QApplication([])
img = np.zeros((160, 220, 4), dtype=np.uint8)
img[..., :3] = 250; img[..., 3] = 255
img[20:140, 20:200, :3] = (30, 60, 200)
img[50:110, 60:160, :3] = 250
img[70:90, 90:130, :3] = (30, 60, 200)
img[30:40, 30:60, :3] = (20, 20, 20)
Image.fromarray(img, "RGBA").save("shot_src.png")

w = MainWindow(); w.resize(1100, 700); w.show()
w.load_image("shot_src.png")
w.set_background_removed(True)
c = w.add_layer(LayerType.COPPER); w.inspector.apply_color((30, 60, 200))
s = w.add_layer(LayerType.SILKSCREEN); w.inspector.apply_color((20, 20, 20))
n = w.generate_negative([c.id], LayerType.MASK)
w.layer_pane.toggle_layer_visibility(s.id)
w.checker_slider.setValue(60)
w.flush_recompute()
w.layer_pane.select_base()
app.processEvents()
w.grab().save("shot_base.png")
w.layer_pane.select_layer(c.id)
app.processEvents()
w.grab().save("shot_layer.png")
w._modified = False
