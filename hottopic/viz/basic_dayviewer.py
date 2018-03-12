# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'hottopic/viz/dayviewer.ui'
#
# Created by: PyQt5 UI code generator 5.10
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_basic_dayviewer(object):
    def setupUi(self, basic_dayviewer):
        basic_dayviewer.setObjectName("basic_dayviewer")
        basic_dayviewer.resize(688, 461)
        self.verticalLayout = QtWidgets.QVBoxLayout(basic_dayviewer)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.AvailableLayers = QtWidgets.QListView(basic_dayviewer)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.AvailableLayers.sizePolicy().hasHeightForWidth())
        self.AvailableLayers.setSizePolicy(sizePolicy)
        self.AvailableLayers.setObjectName("AvailableLayers")
        self.horizontalLayout.addWidget(self.AvailableLayers)
        self.display = QtWidgets.QLabel(basic_dayviewer)
        self.display.setObjectName("display")
        self.horizontalLayout.addWidget(self.display)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.showStartingCheckBox = QtWidgets.QCheckBox(basic_dayviewer)
        self.showStartingCheckBox.setObjectName("showStartingCheckBox")
        self.horizontalLayout_2.addWidget(self.showStartingCheckBox)
        self.showEndingCheckBox = QtWidgets.QCheckBox(basic_dayviewer)
        self.showEndingCheckBox.setObjectName("showEndingCheckBox")
        self.horizontalLayout_2.addWidget(self.showEndingCheckBox)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(basic_dayviewer)
        QtCore.QMetaObject.connectSlotsByName(basic_dayviewer)

    def retranslateUi(self, basic_dayviewer):
        _translate = QtCore.QCoreApplication.translate
        basic_dayviewer.setWindowTitle(_translate("basic_dayviewer", "Dialog"))
        self.display.setText(_translate("basic_dayviewer", "display"))
        self.showStartingCheckBox.setText(_translate("basic_dayviewer", "Starting Perimeter"))
        self.showEndingCheckBox.setText(_translate("basic_dayviewer", "Ending Perimeter"))

