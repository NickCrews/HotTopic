# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hottopic/viz/basicgui.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_GUI(object):
    def setupUi(self, GUI):
        GUI.setObjectName("GUI")
        GUI.setEnabled(True)
        GUI.resize(680, 666)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(GUI.sizePolicy().hasHeightForWidth())
        GUI.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(GUI)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.datasetTab = QtWidgets.QWidget()
        self.datasetTab.setObjectName("datasetTab")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.datasetTab)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox = QtWidgets.QGroupBox(self.datasetTab)
        self.groupBox.setObjectName("groupBox")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.groupBox)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.burnTree = QtWidgets.QTreeView(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.burnTree.sizePolicy().hasHeightForWidth())
        self.burnTree.setSizePolicy(sizePolicy)
        self.burnTree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.burnTree.setIndentation(20)
        self.burnTree.setSortingEnabled(True)
        self.burnTree.setObjectName("burnTree")
        self.burnTree.header().setVisible(False)
        self.burnTree.header().setStretchLastSection(False)
        self.horizontalLayout_3.addWidget(self.burnTree)
        self.burnDisplay = QtWidgets.QLabel(self.groupBox)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.burnDisplay.sizePolicy().hasHeightForWidth())
        self.burnDisplay.setSizePolicy(sizePolicy)
        self.burnDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.burnDisplay.setObjectName("burnDisplay")
        self.horizontalLayout_3.addWidget(self.burnDisplay)
        self.verticalLayout.addWidget(self.groupBox)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.groupBox_2 = QtWidgets.QGroupBox(self.datasetTab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.groupBox_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.label_4 = QtWidgets.QLabel(self.groupBox_2)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_7.addWidget(self.label_4)
        self.datasetDatasetLineEdit = QtWidgets.QLineEdit(self.groupBox_2)
        self.datasetDatasetLineEdit.setReadOnly(True)
        self.datasetDatasetLineEdit.setObjectName("datasetDatasetLineEdit")
        self.horizontalLayout_7.addWidget(self.datasetDatasetLineEdit)
        self.verticalLayout_6.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setContentsMargins(-1, 0, -1, -1)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.loadDatasetButton = QtWidgets.QPushButton(self.groupBox_2)
        self.loadDatasetButton.setObjectName("loadDatasetButton")
        self.horizontalLayout_4.addWidget(self.loadDatasetButton)
        self.saveDatasetButton = QtWidgets.QPushButton(self.groupBox_2)
        self.saveDatasetButton.setObjectName("saveDatasetButton")
        self.horizontalLayout_4.addWidget(self.saveDatasetButton)
        self.verticalLayout_6.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.datasetTree = QtWidgets.QTreeView(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.datasetTree.sizePolicy().hasHeightForWidth())
        self.datasetTree.setSizePolicy(sizePolicy)
        self.datasetTree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.datasetTree.setIndentation(20)
        self.datasetTree.setSortingEnabled(True)
        self.datasetTree.setObjectName("datasetTree")
        self.datasetTree.header().setVisible(False)
        self.datasetTree.header().setStretchLastSection(False)
        self.horizontalLayout_5.addWidget(self.datasetTree)
        self.datasetDisplay = QtWidgets.QLabel(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.datasetDisplay.sizePolicy().hasHeightForWidth())
        self.datasetDisplay.setSizePolicy(sizePolicy)
        self.datasetDisplay.setAlignment(QtCore.Qt.AlignCenter)
        self.datasetDisplay.setObjectName("datasetDisplay")
        self.horizontalLayout_5.addWidget(self.datasetDisplay)
        self.verticalLayout_6.addLayout(self.horizontalLayout_5)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.tabWidget.addTab(self.datasetTab, "")
        self.modelTab = QtWidgets.QWidget()
        self.modelTab.setObjectName("modelTab")
        self.verticalLayout_8 = QtWidgets.QVBoxLayout(self.modelTab)
        self.verticalLayout_8.setObjectName("verticalLayout_8")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.modelLabel = QtWidgets.QLabel(self.modelTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.modelLabel.sizePolicy().hasHeightForWidth())
        self.modelLabel.setSizePolicy(sizePolicy)
        self.modelLabel.setObjectName("modelLabel")
        self.horizontalLayout_2.addWidget(self.modelLabel)
        self.modelLineEdit = QtWidgets.QLineEdit(self.modelTab)
        self.modelLineEdit.setReadOnly(True)
        self.modelLineEdit.setObjectName("modelLineEdit")
        self.horizontalLayout_2.addWidget(self.modelLineEdit)
        self.modelBrowseButton = QtWidgets.QPushButton(self.modelTab)
        self.modelBrowseButton.setObjectName("modelBrowseButton")
        self.horizontalLayout_2.addWidget(self.modelBrowseButton)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.modelDisplay = QtWidgets.QLabel(self.modelTab)
        self.modelDisplay.setObjectName("modelDisplay")
        self.verticalLayout_3.addWidget(self.modelDisplay)
        self.verticalLayout_8.addLayout(self.verticalLayout_3)
        self.tabWidget.addTab(self.modelTab, "")
        self.trainTab = QtWidgets.QWidget()
        self.trainTab.setObjectName("trainTab")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.trainTab)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.formLayout_2.setObjectName("formLayout_2")
        self.modelLabel_3 = QtWidgets.QLabel(self.trainTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.modelLabel_3.sizePolicy().hasHeightForWidth())
        self.modelLabel_3.setSizePolicy(sizePolicy)
        self.modelLabel_3.setObjectName("modelLabel_3")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.modelLabel_3)
        self.trainModelLineEdit = QtWidgets.QLineEdit(self.trainTab)
        self.trainModelLineEdit.setReadOnly(True)
        self.trainModelLineEdit.setObjectName("trainModelLineEdit")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.trainModelLineEdit)
        self.label_5 = QtWidgets.QLabel(self.trainTab)
        self.label_5.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_5.sizePolicy().hasHeightForWidth())
        self.label_5.setSizePolicy(sizePolicy)
        self.label_5.setObjectName("label_5")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.trainDatasetLineEdit = QtWidgets.QLineEdit(self.trainTab)
        self.trainDatasetLineEdit.setReadOnly(True)
        self.trainDatasetLineEdit.setObjectName("trainDatasetLineEdit")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.trainDatasetLineEdit)
        self.verticalLayout_2.addLayout(self.formLayout_2)
        self.tabWidget.addTab(self.trainTab, "")
        self.predictTab = QtWidgets.QWidget()
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictTab.sizePolicy().hasHeightForWidth())
        self.predictTab.setSizePolicy(sizePolicy)
        self.predictTab.setObjectName("predictTab")
        self.formLayout = QtWidgets.QFormLayout(self.predictTab)
        self.formLayout.setObjectName("formLayout")
        self.modelLabel_2 = QtWidgets.QLabel(self.predictTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.modelLabel_2.sizePolicy().hasHeightForWidth())
        self.modelLabel_2.setSizePolicy(sizePolicy)
        self.modelLabel_2.setObjectName("modelLabel_2")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.modelLabel_2)
        self.predictModelLineEdit = QtWidgets.QLineEdit(self.predictTab)
        self.predictModelLineEdit.setReadOnly(True)
        self.predictModelLineEdit.setObjectName("predictModelLineEdit")
        self.formLayout.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.predictModelLineEdit)
        self.label_2 = QtWidgets.QLabel(self.predictTab)
        self.label_2.setEnabled(True)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setObjectName("label_2")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.predictBurnTree = QtWidgets.QTreeView(self.predictTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictBurnTree.sizePolicy().hasHeightForWidth())
        self.predictBurnTree.setSizePolicy(sizePolicy)
        self.predictBurnTree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
        self.predictBurnTree.setIndentation(20)
        self.predictBurnTree.setSortingEnabled(True)
        self.predictBurnTree.setObjectName("predictBurnTree")
        self.predictBurnTree.header().setVisible(False)
        self.predictBurnTree.header().setStretchLastSection(False)
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.predictBurnTree)
        self.predictButton = QtWidgets.QPushButton(self.predictTab)
        self.predictButton.setObjectName("predictButton")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.SpanningRole, self.predictButton)
        self.predictDisplay = QtWidgets.QLabel(self.predictTab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.predictDisplay.sizePolicy().hasHeightForWidth())
        self.predictDisplay.setSizePolicy(sizePolicy)
        self.predictDisplay.setObjectName("predictDisplay")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.predictDisplay)
        self.tabWidget.addTab(self.predictTab, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        GUI.setCentralWidget(self.centralwidget)
        self.actionN = QtWidgets.QAction(GUI)
        self.actionN.setCheckable(True)
        self.actionN.setChecked(True)
        self.actionN.setObjectName("actionN")
        self.actionKg = QtWidgets.QAction(GUI)
        self.actionKg.setCheckable(True)
        self.actionKg.setObjectName("actionKg")
        self.actionLbs = QtWidgets.QAction(GUI)
        self.actionLbs.setCheckable(True)
        self.actionLbs.setObjectName("actionLbs")
        self.actionOpenCal = QtWidgets.QAction(GUI)
        self.actionOpenCal.setShortcut("")
        self.actionOpenCal.setObjectName("actionOpenCal")
        self.actionSaveCal = QtWidgets.QAction(GUI)
        self.actionSaveCal.setObjectName("actionSaveCal")
        self.actionSaveCalAs = QtWidgets.QAction(GUI)
        self.actionSaveCalAs.setObjectName("actionSaveCalAs")
        self.actionSave_2 = QtWidgets.QAction(GUI)
        self.actionSave_2.setObjectName("actionSave_2")
        self.actionOpenRec = QtWidgets.QAction(GUI)
        self.actionOpenRec.setObjectName("actionOpenRec")
        self.actionSaveRec = QtWidgets.QAction(GUI)
        self.actionSaveRec.setObjectName("actionSaveRec")
        self.actionSaveRecAs = QtWidgets.QAction(GUI)
        self.actionSaveRecAs.setObjectName("actionSaveRecAs")
        self.actionExportSnippet = QtWidgets.QAction(GUI)
        self.actionExportSnippet.setObjectName("actionExportSnippet")
        self.actionNew_Dataset = QtWidgets.QAction(GUI)
        self.actionNew_Dataset.setObjectName("actionNew_Dataset")
        self.modelLabel.setBuddy(self.modelBrowseButton)
        self.modelLabel_3.setBuddy(self.modelBrowseButton)
        self.modelLabel_2.setBuddy(self.modelBrowseButton)

        self.retranslateUi(GUI)
        self.tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(GUI)

    def retranslateUi(self, GUI):
        _translate = QtCore.QCoreApplication.translate
        GUI.setWindowTitle(_translate("GUI", "Hot Topic"))
        self.groupBox.setTitle(_translate("GUI", "Available Data:"))
        self.burnDisplay.setText(_translate("GUI", "Select a Burn or Day..."))
        self.groupBox_2.setTitle(_translate("GUI", "Dataset:"))
        self.label_4.setText(_translate("GUI", "Current Dataset:"))
        self.datasetDatasetLineEdit.setText(_translate("GUI", "Not saved..."))
        self.loadDatasetButton.setText(_translate("GUI", "Load..."))
        self.saveDatasetButton.setText(_translate("GUI", "Save..."))
        self.datasetDisplay.setText(_translate("GUI", "Select a Day..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.datasetTab), _translate("GUI", "Dataset"))
        self.modelLabel.setText(_translate("GUI", "Model:"))
        self.modelLineEdit.setText(_translate("GUI", "Not Saved..."))
        self.modelBrowseButton.setText(_translate("GUI", "Browse..."))
        self.modelDisplay.setText(_translate("GUI", "TextLabel"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.modelTab), _translate("GUI", "Model"))
        self.modelLabel_3.setText(_translate("GUI", "Model:"))
        self.trainModelLineEdit.setText(_translate("GUI", "Not Saved..."))
        self.label_5.setText(_translate("GUI", "Dataset: "))
        self.trainDatasetLineEdit.setText(_translate("GUI", "Not Saved..."))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.trainTab), _translate("GUI", "Train"))
        self.modelLabel_2.setText(_translate("GUI", "Model:"))
        self.predictModelLineEdit.setText(_translate("GUI", "Not Saved..."))
        self.label_2.setText(_translate("GUI", "Dataset: "))
        self.predictButton.setText(_translate("GUI", "Predict!"))
        self.predictDisplay.setText(_translate("GUI", "TextLabel"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.predictTab), _translate("GUI", "Predict"))
        self.actionN.setText(_translate("GUI", "N"))
        self.actionKg.setText(_translate("GUI", "kg"))
        self.actionLbs.setText(_translate("GUI", "lbs"))
        self.actionOpenCal.setText(_translate("GUI", "Open..."))
        self.actionSaveCal.setText(_translate("GUI", "Save"))
        self.actionSaveCalAs.setText(_translate("GUI", "Save As..."))
        self.actionSave_2.setText(_translate("GUI", "Save"))
        self.actionOpenRec.setText(_translate("GUI", "Open..."))
        self.actionSaveRec.setText(_translate("GUI", "Save"))
        self.actionSaveRecAs.setText(_translate("GUI", "Save As..."))
        self.actionExportSnippet.setText(_translate("GUI", "Export Snippet..."))
        self.actionNew_Dataset.setText(_translate("GUI", "New Dataset..."))
