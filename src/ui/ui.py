from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QFileDialog, QLabel

# Define a class for the front-end page
class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Real-time Expression Recognition'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 200
        self.initUI()

    def initUI(self):
        # Set the window title and size
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        # Create a label to display the instructions
        self.label = QLabel(self)
        self.label.setText('Please choose one of the following options:')
        self.label.move(50, 20)

        # Create a button to choose the camera option
        self.button1 = QPushButton('Camera', self)
        self.button1.move(50, 80)
        # Connect the button to the camera function
        self.button1.clicked.connect(self.camera)

        # Create a button to choose the video option
        self.button2 = QPushButton('Video', self)
        self.button2.move(250, 80)
        # Connect the button to the video function
        self.button2.clicked.connect(self.video)

        # Show the window
        self.show()

    def camera(self):
        # Set the source argument to 0 for camera
        opt.source = 0
        # Close the window and start the recognition
        self.close()
        predict_expression()

    def video(self):
        # Set the source argument to 1 for video
        opt.source = 1
        # Open a file dialog to choose the video file
        opt.video_path, _ = QFileDialog.getOpenFileName(self, 'Open file', '', 'Video files (*.mp4 *.avi)')
        # Close the window and start the recognition
        print(opt.video_path)
        self.close()
        predict_expression()
