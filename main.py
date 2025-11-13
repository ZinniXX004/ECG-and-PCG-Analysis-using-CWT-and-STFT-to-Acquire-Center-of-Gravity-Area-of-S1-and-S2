"""Start the GUI app."""
import sys
from PyQt6 import QtWidgets
from GUI import PCGAnalyzerGUI

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = PCGAnalyzerGUI()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
