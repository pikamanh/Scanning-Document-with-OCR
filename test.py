from PyQt6.uic import load_ui
from PyQt6.QtWidgets import QApplication
from app.gui.cameraDisplay import CameraDisplay

app = QApplication([])

win = load_ui.loadUi('app/resources/main.ui')

win.show()
app.exec()