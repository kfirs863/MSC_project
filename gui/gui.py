import sys
import numpy as np
import cv2
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog,
    QHBoxLayout, QProgressBar, QMessageBox
)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QRect, QThread, pyqtSignal
from ultralytics import SAM  # Ensure SAM is correctly installed and imported

# Import your utility functions
from tools.utils import display_masked_areas
from tools.orthograohic_image import capture_textured_image_and_depth_from_obj
from tools.extract_and_save_masked_areas import extract_and_save_masked_areas
from tools.generate_topographic_map import generate_topographic_map
from tools.plot_depth_profile_for_mesh import plot_depth_profile_for_mesh
from tools.project_masks_to_3d import project_masks_to_mesh
from tools.process_ply_to_skeleton import process_ply_to_skeleton

class ModelThread(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, model, image_rgb, boxes):
        super().__init__()
        self.model = model
        self.image_rgb = image_rgb
        self.boxes = boxes

    def run(self):
        try:
            self.progress.emit(10)  # Starting
            self.progress.emit(20)
            all_masks = self.model(self.image_rgb, bboxes=self.boxes)
            self.progress.emit(70)
            # Simulate processing steps
            # For example, generating masks could take some time
            # Here, you can add more detailed progress updates if possible
            self.progress.emit(100)  # Finished
            self.finished.emit(all_masks)
        except Exception as e:
            self.error.emit(str(e))


class BoundingBoxApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.image = None
        self.boxes = []
        self.current_rect = None
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.model = SAM("sam2_l.pt").to(torch.device('cpu'))
        self.model.eval()

    def initUI(self):
        self.setWindowTitle('Bounding Box SAM Application')

        layout = QVBoxLayout()

        # File Selection Layout
        file_layout = QHBoxLayout()
        self.load_button = QPushButton('Select .obj File', self)
        self.load_button.clicked.connect(self.load_obj)
        file_layout.addWidget(self.load_button)

        self.process_button = QPushButton('Process with SAM', self)
        self.process_button.clicked.connect(self.process_with_sam)
        self.process_button.setEnabled(False)  # Disabled until boxes are drawn
        file_layout.addWidget(self.process_button)

        layout.addLayout(file_layout)

        # Image Display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignTop)
        self.image_label.mousePressEvent = self.mouse_press_event
        self.image_label.mouseMoveEvent = self.mouse_move_event
        self.image_label.mouseReleaseEvent = self.mouse_release_event
        layout.addWidget(self.image_label)

        # Progress Bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        # Status Label
        self.status_label = QLabel('Status: Idle', self)
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self.setGeometry(100, 100, 800, 600)
        self.show()

    def load_obj(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select .obj File", "", "OBJ Files (*.obj);;All Files (*)", options=options)
        if file_path:
            try:
                self.status_label.setText("Status: Loading OBJ and capturing image...")
                QApplication.processEvents()  # Update UI
                # Load the mesh and capture orthographic image
                image_path, _, _ = capture_textured_image_and_depth_from_obj(
                    file_path, zoom=0.4, number_of_iterations=1, use_sharpen=False, strength=0.1, with_texture=False)

                # Load and display the image
                self.image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if self.image is None:
                    raise ValueError("Failed to load image.")
                self.display_image()
                self.status_label.setText("Status: Image loaded. Draw bounding boxes.")
                self.process_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load OBJ file: {e}")
                self.status_label.setText("Status: Error loading OBJ.")

    def display_image(self):
        if self.image is not None:
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)
            self.image_label.setFixedSize(pixmap.size())
            self.resize(pixmap.width() + 100, pixmap.height() + 150)

    def mouse_press_event(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()

    def mouse_move_event(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouse_release_event(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            rect = QRect(self.start_point, self.end_point).normalized()
            self.boxes.append([rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()])
            self.draw_boxes()
            self.update()

    def draw_boxes(self):
        if self.image is not None:
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB).copy()
            for box in self.boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
            height, width, channel = rgb_image.shape
            bytes_per_line = 3 * width
            q_img = QImage(rgb_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            self.image_label.setPixmap(pixmap)

    def process_with_sam(self):
        if not self.boxes:
            QMessageBox.warning(self, "No Boxes", "Please draw at least one bounding box before processing.")
            return

        try:
            self.status_label.setText("Status: Processing with SAM...")
            self.progress_bar.setValue(0)
            QApplication.processEvents()

            # Convert image to RGB if needed
            image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)

            # Convert boxes to numpy array
            boxes_np = np.array(self.boxes)

            # Initialize and start the model thread
            self.thread = ModelThread(self.model, image_rgb, boxes_np)
            self.thread.progress.connect(self.update_progress)
            self.thread.finished.connect(self.processing_finished)
            self.thread.error.connect(self.processing_error)
            self.thread.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process with SAM: {e}")
            self.status_label.setText("Status: Error during processing.")

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        if value < 100:
            self.status_label.setText(f"Status: Processing... {value}%")
        else:
            self.status_label.setText("Status: Processing completed.")

    def processing_finished(self, all_masks):
        try:
            self.status_label.setText("Status: Processing completed. Displaying masks.")
            self.progress_bar.setValue(100)

            # Handle masks as needed
            # For example, display masks over the image
            for mask in all_masks:
                mask_image = mask.show()  # Assuming mask.show() returns a displayable image
                # Convert mask to QImage and display or overlay on the original image
                mask_np = mask_image.numpy().astype(np.uint8) * 255  # Example conversion
                mask_color = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(self.image, 0.7, mask_color, 0.3, 0)
                rgb_overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                height, width, channel = rgb_overlay.shape
                bytes_per_line = 3 * width
                q_img = QImage(rgb_overlay.data, width, height, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_img)
                self.image_label.setPixmap(pixmap)
                self.image_label.setFixedSize(pixmap.size())

            # Further processing steps can be called here
            # e.g., saving masks, projecting onto mesh, etc.

            QMessageBox.information(self, "Success", "SAM processing completed successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error displaying masks: {e}")
            self.status_label.setText("Status: Error displaying masks.")

    def processing_error(self, error_message):
        QMessageBox.critical(self, "Processing Error", f"An error occurred: {error_message}")
        self.status_label.setText("Status: Error during processing.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = BoundingBoxApp()
    sys.exit(app.exec_())
