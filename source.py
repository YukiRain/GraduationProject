import PyQt5.uic as pyuic
from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PIL import Image
import ImageFusion as fus
import sys,os

class MainWin(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWin,self).__init__()
        self.ui=pyuic.loadUi("fusionWindow.ui",self)
        self.ui.openFilesButton.clicked.connect(self.OpenImg)
        self.ui.optionButton.clicked.connect(self.option)
        self.ui.fusionButton.clicked.connect(self.fusion)
        self.__setting={'average':True,'sqlSave':False,'axis on':False,'imgSave':False}
        self.__win=[]

    def OpenImg(self):
        files,tp=QtWidgets.QFileDialog.getOpenFileNames(self,
                                                        "Open Images",
                                                        "F:",
                                                        "(*);;(*.jpg);;(*.png);;(*.bmp);;(*.jpeg)")
        img=list(filter(self.__imgFilter,files))
        if img:
            self.ui.statusBar.showMessage("Opened "+str(len(img))+" Files.")
            self.picLabel=[]
            for i in img:
                label=QtWidgets.QLabel()
                label.setText(i)
                self.picLabel.append(label)
            for p in self.picLabel:
                pic=QPixmap(p.text())
                p.resize(120,100)
                self.ui.HL1.addWidget(p)
                pic=pic.scaled(p.size(),QtCore.Qt.IgnoreAspectRatio)
                p.setPixmap(pic)
        else:
            self.ui.statusBar.showMessage("Opened No File.")
        self.__win=img

    def option(self):
        self.myOption=QtWidgets.QDialog()
        self.ui2=pyuic.loadUi('optionWindow.ui',self.myOption)
        self.ui2.cancelButton.clicked.connect(self.myOption.close)
        self.ui2.okButton.clicked.connect(self.setting)
        self.myOption.show()

    def fusion(self):
        if not self.__win:
            QtWidgets.QMessageBox.warning(self,
                                          'FBI Warning',
                                          '未打开文件',
                                          QtWidgets.QMessageBox.Cancel)
        else:
            origin = fus.openFiles(self.__win)
            af = fus.averageFusion(origin)
            fus.img_show(origin, af, self.__setting['axis on'])
            save = Image.fromarray(af)
            save.save('F:\\GraduationProject\\Source\\cache\\1.jpg')
            label = QtWidgets.QLabel()
            label.resize(120, 100)
            self.ui.HL2.addWidget(label)
            pic = QPixmap('F:\\GraduationProject\\Source\\cache\\1.jpg')
            pic = pic.scaled(label.size(), QtCore.Qt.IgnoreAspectRatio)
            label.setPixmap(pic)
            if not self.__setting['imgSave']:
                os.remove('F:\\GraduationProject\\Source\\cache\\1.jpg')

    def setting(self):
        avr = self.ui2.avr_check.checkState()
        axis = self.ui2.axis_check.checkState()
        save = self.ui2.save_check.checkState()
        sql = self.ui2.sql_check.checkState()
        self.__setting['average'] = True if avr == QtCore.Qt.Checked else False
        self.__setting['sqlSave'] = True if sql == QtCore.Qt.Checked else False
        self.__setting['axis on'] = True if axis == QtCore.Qt.Checked else False
        self.__setting['imgSave'] = True if save == QtCore.Qt.Checked else False
        self.myOption.close()
        self.ui.statusBar.showMessage('Average:'+str(self.__setting['average'])+
                                      ';  SQL:'+str(self.__setting['sqlSave'])+
                                      ';  Axis:'+str(self.__setting['axis on'])+
                                      ';  Save:'+str(self.__setting['imgSave']))

    def __imgFilter(self,path):
        suffix=path.split('.')
        return suffix[len(suffix)-1].upper()=='JPG' or\
               suffix[len(suffix)-1].upper()=='PNG' or\
               suffix[len(suffix)-1].upper()=='BMP' or\
               suffix[len(suffix)-1].upper()=='JPEG' or\
               suffix[len(suffix)-1].upper()=='WMF'


if __name__=='__main__':
    app=QtWidgets.QApplication(sys.argv)
    myshow=MainWin()
    myshow.show()
    sys.exit(app.exec_())