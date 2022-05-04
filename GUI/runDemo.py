"""
 @Time    : 2021/5/2 7:55
 @Author  : Monlter
 @FileName: runDemo.py
 @Software: PyCharm
"""
import sys
from surface import Ui_MainWindow
from PyQt5.QtWidgets import QApplication, QWidget,QMainWindow


class mywindow(QMainWindow):
    def __init__(self):
        super(mywindow, self).__init__()
        self.new = Ui_MainWindow()
        self.new.setupUi(self)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myshow = mywindow()
    myshow.show()
    sys.exit(app.exec_())
