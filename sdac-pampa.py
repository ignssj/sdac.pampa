
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from Unet.unet_model import UNet
from dataset import MyLidcDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

OUTPUT_DIR = "output/"
PATIENT_ID = ""
INPUT_PATIENT = ""
PATIENT_IMG_PATHS = ""

class Ui_MainFrame(object):
    def setupUi(self, MainFrame):
        MainFrame.setObjectName("MainFrame")
        MainFrame.resize(592, 425)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\/Icons/unipampa-icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainFrame.setWindowIcon(icon)
        MainFrame.setLocale(QtCore.QLocale(QtCore.QLocale.Portuguese, QtCore.QLocale.Brazil))
        self.pushButton = QtWidgets.QPushButton(MainFrame)
        self.pushButton.setGeometry(QtCore.QRect(120, 320, 161, 41))
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(".\\/Icons/lupa.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton.setIcon(icon1)
        self.pushButton.setIconSize(QtCore.QSize(18, 18))
        self.pushButton.setObjectName("pushButton")
        self.textEdit = QtWidgets.QTextEdit(MainFrame)
        self.textEdit.setEnabled(True)
        self.textEdit.setGeometry(QtCore.QRect(80, 100, 441, 101))
        self.textEdit.setAutoFillBackground(False)
        self.textEdit.setReadOnly(True)
        self.textEdit.setObjectName("textEdit")
        self.pushButton_2 = QtWidgets.QPushButton(MainFrame)
        self.pushButton_2.setGeometry(QtCore.QRect(320, 320, 161, 41))
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(".\\/Icons/diagnosis.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_2.setIcon(icon2)
        self.pushButton_2.setIconSize(QtCore.QSize(20, 20))
        self.pushButton_2.setObjectName("pushButton_2")

        self.retranslateUi(MainFrame)
        QtCore.QMetaObject.connectSlotsByName(MainFrame)


    def open_folder_dialog(self):
        # Abre o diálogo de seleção de pasta e armazena o caminho da pasta selecionada
        folder_path = QFileDialog.getExistingDirectory()

        # Se uma pasta for selecionada, crie uma lista contendo os caminhos para cada slice
        if folder_path:
            global INPUT_PATIENT 
            INPUT_PATIENT = folder_path
            PATIENT_MASK = INPUT_PATIENT.replace("Image","Mask")


            input_slices = os.listdir(INPUT_PATIENT)
            full_image_paths = [os.path.join(INPUT_PATIENT, slice) for slice in input_slices]

            global PATIENT_IMG_PATHS
            PATIENT_IMG_PATHS = full_image_paths

            mask_slices = os.listdir(PATIENT_MASK)
            full_mask_paths = [os.path.join(PATIENT_MASK, slice) for slice in mask_slices]

            list_image_paths = list(full_image_paths)
            list_mask_paths = list(full_mask_paths)

            # extrai o ID do paciente
            global PATIENT_ID
            PATIENT_ID = os.path.basename(os.path.dirname(full_image_paths[0]))

            # Directory to save U-Net predict output
            print("Salvando resultados em {}".format(OUTPUT_DIR+PATIENT_ID))
            os.makedirs(OUTPUT_DIR+PATIENT_ID,exist_ok=True)

            test_dataset = MyLidcDataset(list_image_paths, list_mask_paths)
            test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=6)

            counter = 0
            with torch.no_grad():

                for input, target in test_loader:
                    input = input.cuda()
                    target = target.cuda()

                    output = model(input)



                    output = torch.sigmoid(output)
                    output = (output>0.5).float().cpu().numpy()
                    output = np.squeeze(output,axis=1)

                    for i in range(output.shape[0]):
                        label = list_image_paths[counter][-23:]
                        label = label.replace('NI','PD')
                        np.save(OUTPUT_DIR+PATIENT_ID+'/'+label,output[i,:,:])
                        counter+=1

    def open_result_folder(self):
        folder_path = QFileDialog.getExistingDirectory()
        # Se uma pasta for selecionada, crie uma lista contendo os caminhos para cada slice
        if folder_path:
            outputs = os.listdir(folder_path)
            full_result_paths = [os.path.join(folder_path, output) for output in outputs]

            fig, axs = plt.subplots(1, len(full_result_paths))
            fig.suptitle('Nodulos suspeitos encontrados para o paciente '+PATIENT_ID,fontsize=30,fontweight='bold')

            for i, array_file in enumerate(full_result_paths):
                original_ct = np.load(PATIENT_IMG_PATHS[i])
                detected_nodule = np.load(array_file)
                axs[i].imshow(original_ct+detected_nodule)
                axs[i].set_title(os.path.basename(array_file).split(".")[0])
                
            plt.show()
            plt.get_current_fig_manager().window.showMaximized()

    def retranslateUi(self, MainFrame):
        _translate = QtCore.QCoreApplication.translate
        MainFrame.setWindowTitle(_translate("MainFrame", "SDAC.PAMPA"))
        self.pushButton.setText(_translate("MainFrame", "Detectar Nódulos"))
        self.pushButton.clicked.connect(self.open_folder_dialog)
        self.textEdit.setHtml(_translate("MainFrame", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">Seja bem-vindo ao Sistema de Diagnostico Assistido por Computador </span><span style=\" font-family:\'Ubuntu\'; font-size:11pt; font-weight:600;\">SDAC.PAMPA</span><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">!</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-family:\'Ubuntu\'; font-size:11pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-family:\'Ubuntu\'; font-size:11pt;\">Para prosseguir com o uso do software, selecione uma tomografia a partir do botão Detectar Nodulos.</span></p></body></html>"))
        self.pushButton_2.setText(_translate("MainFrame", "Mostrar Resultado"))
        self.pushButton_2.clicked.connect(self.open_result_folder)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainFrame()
    ui.setupUi(MainWindow)
    MainWindow.show()

    with open('E:\Projetos\Python\sdac.pampa/config.yml', 'r') as f:
        config = yaml.safe_load(f)
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    state_dict = torch.load("E:\Projetos\Python\sdac.pampa/model.pth")
    model.load_state_dict(state_dict['model_state_dict'])
    model = model.cuda()


    sys.exit(app.exec_())