# +----------------------------------------------------------------------------+
# | Machine Learning with TEXT DATA                                            |
# +----------------------------------------------------------------------------+
# |                                                                            |
# |                                                                            |
# |                                                                            |
# +----------------------------------------------------------------------------+

import csv
import os
import io
import sys
import random
import time
import traceback
import numpy as np
import configparser
from sys import platform as _platform

from PyQt5 import *
from PyQt5 import uic, QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
import pandas as pd
import string
import nltk
import sklearn
from nltk.corpus import wordnet as wn

__version__ = "1.0"
__programname__ = "ML_TEXT"
__emailaddress__ = ""

darkthemeavailable = 1
try:
    import qdarkstyle
except ModuleNotFoundError:
    darkthemeavailable = 0


def resource_path(relative_path):  # Define function to import external files when using PyInstaller.
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# GUI layout files.
qtmainfile = resource_path("UI/mainwindow.ui")
Ui_main, QtBaseClass = uic.loadUiType(qtmainfile)
qthelpfile = resource_path("UI/help.ui")
Ui_help, QtBaseClass = uic.loadUiType(qthelpfile)
qtsettingsfile = resource_path("UI/settings.ui")
Ui_settings, QtBaseClass = uic.loadUiType(qtsettingsfile)
qtintrofile = resource_path("UI/intro.ui")
Ui_intro, QtBaseClass = uic.loadUiType(qtintrofile)
qtresultfile = resource_path("UI/result.ui")
Ui_result, QtBaseClass = uic.loadUiType(qtresultfile)
qtdatafile = resource_path("UI/data_window.ui")
Ui_data, QtBaseClass = uic.loadUiType(qtdatafile)
qtmodelfile = resource_path("UI/select_model.ui")
Ui_model, QtBaseClass = uic.loadUiType(qtmodelfile)

config = configparser.ConfigParser()
config.read(resource_path('configuration.ini'))
show_intro = int(config["Settings"]["show_intro"])
colortheme = int(config["Settings"]["colortheme"])
window_start_view = int(config["Settings"]["window_start_view"])

tag_map = {
        'CC':None, # coordin. conjunction (and, but, or)
        'CD':wn.NOUN, # cardinal number (one, two)
        'DT':None, # determiner (a, the)
        'EX':wn.ADV, # existential ‘there’ (there)
        'FW':None, # foreign word (mea culpa)
        'IN':wn.ADV, # preposition/sub-conj (of, in, by)
        'JJ':wn.ADJ, # adjective (yellow)
        'JJR':wn.ADJ, # adj., comparative (bigger)
        'JJS':wn.ADJ, # adj., superlative (wildest)
        'LS':None, # list item marker (1, 2, One)
        'MD':None, # modal (can, should)
        'NN':wn.NOUN, # noun, sing. or mass (llama)
        'NNS':wn.NOUN, # noun, plural (llamas)
        'NNP':wn.NOUN, # proper noun, sing. (IBM)
        'NNPS':wn.NOUN, # proper noun, plural (Carolinas)
        'PDT':wn.ADJ_SAT, # predeterminer (all, both)
        'POS':None, # possessive ending (’s )
        'PRP':None, # personal pronoun (I, you, he)
        'PRP$':None, # possessive pronoun (your, one’s)
        'RB':wn.ADV, # adverb (quickly, never)
        'RBR':wn.ADV, # adverb, comparative (faster)
        'RBS':wn.ADV, # adverb, superlative (fastest)
        'RP':wn.ADJ, # particle (up, off)
        'SYM':None, # symbol (+,%, &)
        'TO':None, # “to” (to)
        'UH':None, # interjection (ah, oops)
        'VB':wn.VERB, # verb base form (eat)
        'VBD':wn.VERB, # verb past tense (ate)
        'VBG':wn.VERB, # verb gerund (eating)
        'VBN':wn.VERB, # verb past participle (eaten)
        'VBP':wn.VERB, # verb non-3sg pres (eat)
        'VBZ':wn.VERB, # verb 3sg pres (eats)
        'WDT':None, # wh-determiner (which, that)
        'WP':None, # wh-pronoun (what, who)
        'WP$':None, # possessive (wh- whose)
        'WRB':None, # wh-adverb (how, where)
    }


class help_GUI(QDialog, Ui_help):
    """Documentation/help window."""

    def __init__(self, root):
        QDialog.__init__(self, root)
        Ui_help.__init__(self)
        self.setupUi(self)

        self.currentindex = 0
        self.numberofpages = 0

        self.stackedWidget.removeWidget(self.page_2)
        self.stackedWidget.setCurrentIndex(self.currentindex)

        if colortheme + darkthemeavailable < 2:
            logo = QPixmap(resource_path('Images/logo_b.png'))
        else:
            logo = QPixmap(resource_path('Images/logo_w.png'))
        logo = logo.scaled(180, 60, transformMode=Qt.SmoothTransformation)
        self.lb_logo.setPixmap(logo)

        self.textEdit.setStyleSheet("background: rgba(0,0,255,0%)")


class settings_GUI(QDialog, Ui_settings):
    """Customized settings window including hyperparameters."""

    def __init__(self, root):
        QDialog.__init__(self, root)
        Ui_settings.__init__(self)
        self.setupUi(self)
        if _platform == "darwin":
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        x = app.desktop().screenGeometry().center().x()
        y = app.desktop().screenGeometry().center().y()
        self.move(x - self.geometry().width() // 2, y - self.geometry().height() // 2)

        self.root = root
        self.status = []
        self.statuschanged = 0

        self.status.append(int(config["Settings"]["colortheme"]))
        self.status.append(int(config["Settings"]["window_start_view"]))

        self.themes = ["Classic", "Dark", "Pure Black"]
        self.choice_theme.addItems(self.themes)
        self.choice_theme.setCurrentIndex(int(config["Settings"]["colortheme"]))
        self.views = ["Full Screen", "Maximized", "Normal"]
        self.choice_views.addItems(self.views)
        self.choice_views.setCurrentIndex(int(config["Settings"]["window_start_view"]) - 1)

        self.resultbox.accepted.connect(self.buttonOkayfuncton)
        self.resultbox.rejected.connect(self.buttonCancelfuncton)
        self.shortcut = QShortcut(QtGui.QKeySequence(Qt.Key_Enter), self)
        self.shortcut.activated.connect(self.buttonOkayfuncton)

        # Hyperparameters
        self.choice1.addItems(root.distance_metrics)
        self.choice1.setCurrentIndex(root.distance_metrics.index(root.para_knn["metric"]))
        self.choice2.addItems(root.kernels)
        self.choice2.setCurrentIndex(root.kernels.index(root.para_svm["kernel"]))
        self.choice3.addItems(root.solvers)
        self.choice3.setCurrentIndex(root.solvers.index(root.para_nn["solver"]))
        self.choice4.addItems(root.activations)
        self.choice4.setCurrentIndex(root.activations.index(root.para_nn["activation"]))

        getattr(self, "n_neighbors").setText(str(root.para_knn["n_neighbors"]))
        for para in root.para_svm:
            if para != "kernel":
                getattr(self, para).setText(str(root.para_svm[para]))
        getattr(self, "num_layers").setText(str(len(root.para_nn["hidden_layer_sizes"])))
        getattr(self, "num_nodes").setText(str(root.para_nn["hidden_layer_sizes"][0]))
        getattr(self, "alpha").setText(str(root.para_nn["alpha"]))
        getattr(self, "learning_rate_init").setText(str(root.para_nn["learning_rate_init"]))
        getattr(self, "max_iter2").setText(str(root.para_nn["max_iter"]))
        getattr(self, "random_state2").setText(str(root.para_nn["random_state"]))

    def buttonOkayfuncton(self):
        cfgfile = open(resource_path('configuration.ini'), 'w')

        if self.choice_theme.currentIndex() != self.status[0] or self.choice_views.currentIndex() != self.status[1]:
            self.statuschanged = 1
        config.set("Settings", "colortheme", str(self.choice_theme.currentIndex()))
        config.set("Settings", "window_start_view", str(self.choice_views.currentIndex() + 1))
        config.write(cfgfile)
        cfgfile.close()

        # Hypermaparameters
        self.root.para_knn["metric"] = self.root.distance_metrics[self.choice1.currentIndex()]
        self.root.para_svm["kernel"] = self.root.kernels[self.choice2.currentIndex()]
        self.root.para_nn["solver"] = self.root.solvers[self.choice3.currentIndex()]
        self.root.para_nn["activation"] = self.root.activations[self.choice4.currentIndex()]

        self.root.para_knn["n_neighbors"] = int(getattr(self, "n_neighbors").text())
        self.root.para_svm["C"] = float(getattr(self, "C").text())
        self.root.para_svm["degree"] = int(getattr(self, "degree").text())
        self.root.para_svm["gamma"] = float(getattr(self, "gamma").text()) if getattr(self, "gamma").text() != "auto" \
            else getattr(self, "gamma").text()
        self.root.para_svm["max_iter"] = int(getattr(self, "max_iter").text())
        self.root.para_svm["random_state"] = \
            int(getattr(self, "random_state").text()) if getattr(self, "random_state").text() != "None" else None
        self.root.para_nn["hidden_layer_sizes"] = \
            [int(getattr(self, "num_nodes").text()) for _ in range(int(getattr(self, "num_layers").text()))]
        self.root.para_nn["alpha"] = float(getattr(self, "alpha").text())
        self.root.para_nn["learning_rate_init"] = float(getattr(self, "learning_rate_init").text())
        self.root.para_nn["max_iter"] = int(getattr(self, "max_iter2").text())
        self.root.para_nn["random_state"] = \
            int(getattr(self, "random_state2").text()) if getattr(self, "random_state2").text() != "None" else None

    def buttonCancelfuncton(self):
        pass


class loadmodel(QDialog, Ui_model):

    """Load existing layer structure window."""

    def __init__(self, root, modellist, default_model):
        QDialog.__init__(self, root)
        Ui_model.__init__(self)
        self.setupUi(self)
        if _platform == "darwin":
            self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        x = app.desktop().screenGeometry().center().x()
        y = app.desktop().screenGeometry().center().y()
        self.move(x - self.geometry().width() // 2, y - self.geometry().height() // 2-50)

        self.filelist = modellist
        self.index1 = self.filelist.index(default_model)
        self.result = -1

        self.modeloption.addItems(self.filelist)
        self.modeloption.setCurrentIndex(self.index1)
        self.modeloption.currentIndexChanged.connect(self.selectionchange)
        self.resultbox.accepted.connect(self.buttonOkayfuncton)
        self.resultbox.rejected.connect(self.buttonCancelfuncton)
        self.shortcut = QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Enter), self)
        self.shortcut.activated.connect(self.buttonOkayfuncton)

    def buttonOkayfuncton(self):
        self.result = self.index1

    def buttonCancelfuncton(self):
        self.result = -1

    def returnresult(self):
        return self.result

    def selectionchange(self, i):
        self.index1 = i


class intro_GUI(QDialog, Ui_intro):
    """Introduction window."""

    def __init__(self, root):
        QDialog.__init__(self, root)
        Ui_intro.__init__(self)
        self.setupUi(self)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        x = app.desktop().screenGeometry().center().x()
        y = app.desktop().screenGeometry().center().y()
        self.move(x - self.geometry().width() // 2, y - self.geometry().height() // 2)

        logo = QPixmap(resource_path('Images/logo_w.png'))
        logo = logo.scaled(180, 60, transformMode=Qt.SmoothTransformation)
        self.lb_logo.setPixmap(logo)
        self.lb_logo.setStyleSheet("color: white; background: rgba(0,0,255,0%)")
        self.textEdit.setStyleSheet("color: white; background: rgba(0,0,255,0%)")
        self.frame.setStyleSheet("color: white; background: rgba(0,0,255,0%)")
        self.frame2.setStyleSheet("color: white; background: rgba(0,0,255,0%)")
        self.lb_1.setStyleSheet("color: white; background: rgba(0,0,255,0%)")
        self.checkBox.setStyleSheet("color: white; background: rgba(0,0,255,0%)")

        self.status = []
        self.statuschanged = 0

        self.status.append(int(config["Settings"]["colortheme"]))
        self.status.append(int(config["Settings"]["window_start_view"]))

        self.themes = ["Classic", "Dark", "Pure Black"]
        self.choice_theme.addItems(self.themes)
        self.choice_theme.setCurrentIndex(int(config["Settings"]["colortheme"]))
        self.views = ["Full Screen", "Maximized", "Normal"]
        self.choice_views.addItems(self.views)
        self.choice_views.setCurrentIndex(int(config["Settings"]["window_start_view"]) - 1)
        self.choice_theme.setStyleSheet('selection-background-color: rgb(168,168,168)')
        self.choice_views.setStyleSheet('selection-background-color: rgb(168,168,168)')

        self.resultbox.accepted.connect(self.buttonOkayfuncton)
        self.shortcut = QShortcut(QtGui.QKeySequence(Qt.Key_Enter), self)
        self.shortcut.activated.connect(self.buttonOkayfuncton)

    def buttonOkayfuncton(self):
        cfgfile = open(resource_path('configuration.ini'), 'w')

        if self.choice_theme.currentIndex() != self.status[0] or self.choice_views.currentIndex() != self.status[1]:
            self.statuschanged = 1
        config.set("Settings", "colortheme", str(self.choice_theme.currentIndex()))
        config.set("Settings", "window_start_view", str(self.choice_views.currentIndex() + 1))
        if self.checkBox.isChecked():
            config.set("Settings", "show_intro", "0")
        config.write(cfgfile)
        cfgfile.close()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), QPixmap(resource_path('Images/bg.png')))
        QDialog.paintEvent(self, event)


class result_GUI(QDialog, Ui_result):
    """Show fitting result window."""

    def __init__(self, root):
        QDialog.__init__(self, root)
        Ui_result.__init__(self)
        self.setupUi(self)
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        x = app.desktop().screenGeometry().center().x()
        y = app.desktop().screenGeometry().center().y()
        self.move(x - self.geometry().width() // 2, y - self.geometry().height() // 2)

        if root.fitobject:
            self.label_1.setText("{:.6f}".format(root.fitobject.score_10FCV))
            self.label_2.setText("{:.6f}".format(root.fitobject.score_10FCV_std))
            self.label_3.setText("{:.6f}".format(root.fitobject.score))

    def buttonOkayfuncton(self):
        pass


class data_GUI(QDialog, Ui_data):
    """Data window."""

    def __init__(self, root, file):
        QDialog.__init__(self, root)
        Ui_data.__init__(self)
        self.setupUi(self)
        self.setWindowTitle(file)
        x = app.desktop().screenGeometry().center().x()
        y = app.desktop().screenGeometry().center().y()
        self.move(x - self.geometry().width() // 2, y - self.geometry().height() // 2)

        df = pd.read_csv(file, na_filter=False)
        col1 = df.iloc[:, 0].values
        col2 = df.iloc[:, 1].values
        self.table.setRowCount(len(col1))
        self.table.setHorizontalHeaderLabels([df.columns[0], df.columns[1]])
        for i in range(len(col1)):
            self.table.setItem(i, 0, QTableWidgetItem(str(col1[i])))
            self.table.setItem(i, 1, QTableWidgetItem(col2[i]))
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.move(0, 0)


class SystemTrayIcon(QSystemTrayIcon):
    """A system tray icon can be used to exit the program or show/minimized the window. """

    def __init__(self, icon, parent, window):
        QSystemTrayIcon.__init__(self, icon, parent)
        self.parent = parent
        self.window = window
        self.menu = QMenu()
        self.exitAction = self.menu.addAction("Exit")
        self.exitAction.triggered.connect(self.quitmainwindow)
        self.setContextMenu(self.menu)
        if _platform != "darwin":
            self.activated.connect(self.__icon_activated)

        self.window_showing = 1
        self.window_status = 1
        self.show()

    def quitmainwindow(self):
        self.hide()
        self.deleteLater()
        sys.exit()
        # self.parent.close()

    if _platform != "darwin":
        def __icon_activated(self, reason):
            if reason == QSystemTrayIcon.DoubleClick:
                if self.window_showing == 0:
                    if self.window_status == 1:
                        self.window.showFullScreen()
                    elif self.window_status == 2:
                        self.window.showMaximized()
                    else:
                        self.window.showNormal()
                    self.window_showing = 1
                elif self.window_showing == 1:
                    if self.window.isFullScreen():
                        self.window_status = 1
                    elif self.window.isMaximized():
                        self.window_status = 2
                    else:
                        self.window_status = 3
                    self.window.showMinimized()
                    self.window_showing = 0
                self.window.show()


class ThreadSignals(QObject):

    """Defines the signals available from a running worker thread."""

    finished = pyqtSignal()
    error = pyqtSignal(tuple)
    result = pyqtSignal(object)
    progress = pyqtSignal(tuple)


class Worker(QRunnable):

    """
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    """

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = ThreadSignals()

        # Add the callback to our kwargs
        kwargs['progress_callback'] = self.signals.progress

    @pyqtSlot()
    def run(self):

        """Initialise the runner function with passed args, kwargs."""

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class LoadTEXTDatabase:
    """Main text data processor."""

    def __init__(self, filename, model, parameters, root, progress_callback):
        self.filename = filename
        self.model = model
        self.para = parameters
        self.root = root
        self.progress_callback = progress_callback

        self.score_10FCV = 0
        self.score_10FCV_std = 0
        self.score = 0
        self.initialize()

    def initialize(self):
        self.step(5, "Loading nltk...", lambda: self.load_nltk())
        records = self.step(10, "Reading csv file...", lambda: pd.read_csv(self.filename, na_filter=False))
        processed_records = self.step(15, "Cleaning data...", lambda: self.process_all(records))
        rare_words = self.step(35, "Collecting rare words...", lambda: self.get_rare_words(processed_records))
        (self.tfidf, X) = self.step(40, "Creating the feature matrix...", lambda: self.create_features(processed_records, rare_words))
        y = self.step(60, "Creating the result array...", lambda: self.create_labels(processed_records))
        self.X_train, self.X_valid, self.y_train, self.y_valid = self.step(65, "Splitting train and test data set...",
                                                                           lambda: train_test_split(X, y, test_size=0.2, random_state=0))
        self.step(70, "Training the model...", lambda: self.training())
        self.step(80, "Testing the model...", lambda: self.testing())
        self.progress_callback.emit((90, "Done"))

    def step(self, percentage, description, f):
        if self.root.abortmission == 0:
            self.progress_callback.emit((percentage, description))
            return f()

    def load_nltk(self):
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        # Verify that the following commands work for you, before moving on.
        lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
        stopwords = nltk.corpus.stopwords.words('english')

    def training(self):
        self.classifier = self.learn_classifier(self.X_train, self.y_train, self.model, self.para)

    def learn_classifier(self, X_train, y_train, modelname, para):
        """ learns a classifier from the input features and labels using the model selected and parameters provided."""

        if modelname == "kNN":
            model = KNeighborsClassifier(**para)
        elif modelname == "SVM":
            model = SVC(**para)
        elif modelname == "Neural Network":
            model = MLPClassifier(**para)
        else:
            model = KNeighborsClassifier(**para)

        model.fit(X_train, y_train)
        return model

    def testing(self):
        self.score_10FCV, self.score_10FCV_std, self.score \
            = self.evaluate_classifier(self.classifier, self.X_train, self.y_train, self.X_valid, self.y_valid)

    def evaluate_classifier(self, classifier, X_train, y_train, X_validation, y_validation):
        """ evaluates a classifier based on a supplied train/validation data"""
        score1 = cross_val_score(classifier, X_train, y_train, cv=10)
        score2 = classifier.score(X_validation, y_validation)
        return score1.mean(), score1.std(), score2

    def predicting(self):
        unlabeled_records = pd.read_csv("records_test.csv", na_filter=False)
        y_pred = self.classify_records(self.tfidf, self.classifier, unlabeled_records)

    def mirror_tokenizer(self, x):
        return x

    def process(self, text, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
        """ Normalizes case and handles punctuation
            Inputs:
                text: str: raw text
                lemmatizer: an instance of a class implementing the lemmatize() method
                            (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
            Outputs:
                list(str): tokenized text
            """
        text = text.lower()
        text = text.replace("-", " ").replace("'s", "").replace("'", "")
        # text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
        text = nltk.word_tokenize(text)
        tokens = nltk.pos_tag(text)
        tokens = [token for token in tokens if token[1] in tag_map and tag_map[token[1]]]

        def get_wordnet_pos(treebank_tag):
            if tag_map[treebank_tag]:
                return tag_map[treebank_tag]
            else:
                print("Error: Could not find token tap in tag_map.")
                return

        tokens = [lemmatizer.lemmatize(token[0], get_wordnet_pos(token[1])) for token in tokens]
        return tokens

    def process_all(self, df, lemmatizer=nltk.stem.wordnet.WordNetLemmatizer()):
        """ process all text in the dataframe using process_text() function.
            Inputs
                df: pd.DataFrame: dataframe containing a column 'text' loaded from the CSV file
                lemmatizer: an instance of a class implementing the lemmatize() method
                            (the default argument is of type nltk.stem.wordnet.WordNetLemmatizer)
            Outputs
                pd.DataFrame: dataframe in which the values of text column have been changed from str to list(str),
                                the output from process_text() function. Other columns are unaffected.
            """
        df["text"] = df["text"].apply(lambda record: self.process(record))
        return df

    def get_rare_words(self, processed_records):
        """ use the word count information across all records in training data to come up with a feature list
            Inputs:
                processed_records: pd.DataFrame: the output of process_all() function
            Outputs:
                list(str): list of rare words, sorted alphabetically.
            """
        word_dic = {}
        for tweet in processed_records["text"]:
            for word in tweet:
                word_dic[word] = word_dic[word] + 1 if word in word_dic else 1
        list_of_rare_words = [word for word in word_dic if word_dic[word] == 1]
        return list_of_rare_words

    def create_features(self, processed_records, rare_words):
        """ creates the feature matrix using the processed tweet text
            Inputs:
                records: pd.DataFrame: records read from train/test csv file, containing the column 'text'
                rare_words: list(str): one of the outputs of get_feature_and_rare_words() function
            Outputs:
                sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used
                                                        we need this to tranform test records in the same way as train records
                scipy.sparse.csr.csr_matrix: sparse bag-of-words TF-IDF feature matrix
            """
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        # filtered_words = [word for word in word_dic if not word in rare_words]
        # filtered_words = [word for word in filtered_words if not word in stop_words]
        corpus = []
        for tweet in processed_records["text"]:
            filtered_words = [word for word in tweet if word not in rare_words]
            filtered_words = [word for word in filtered_words if word not in stop_words]
            # tweet_doc = " ".join(filtered_words)
            corpus.append(filtered_words)
        vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(tokenizer=self.mirror_tokenizer, lowercase=False)
        X = vectorizer.fit_transform(corpus)
        # print(len(vectorizer.get_feature_names()))    # 8386
        # print(X.shape)    # (17298, 8386)
        # print(type(X))    # <class 'scipy.sparse.csr.csr_matrix'>
        return vectorizer, X

    def create_labels(self, processed_records):
        """ creates the class labels from screen_name
            Inputs:
                records: pd.DataFrame: records read from train file, containing the column 'screen_name'
            Outputs:
                numpy.ndarray(int): dense binary numpy array of class labels
            """
        # Check if the csv file is valid
        cols = [col for col in processed_records.columns]
        if len(cols) != 2:
            return None
        else:
            cols = [col for col in cols if col != "text"]
            col = cols[0]

        list_of_names = []
        if col == "screen_name":
            for name in processed_records[col]:
                if name in ["realDonaldTrump", "mike_pence", "GOP"]:
                    list_of_names.append(0)
                else:
                    list_of_names.append(1)
        else:
            for label in processed_records[col]:
                list_of_names.append(int(label))
        return np.array(list_of_names)

    def classify_records(self, tfidf, classifier, unlabeled_records):
        """ predicts class labels for raw tweet text
            Inputs:
                tfidf: sklearn.feature_extraction.text.TfidfVectorizer: the TfidfVectorizer object used on training data
                classifier: sklearn.svm.classes.SVC: classifier learnt
                unlabeled_records: pd.DataFrame: records read from records_test.csv
            Outputs:
                numpy.ndarray(int): dense binary vector of class labels for unlabeled records
            """
        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))
        processed_unlabeled_records = self.process_all(unlabeled_records)
        corpus = []
        for tweet in processed_unlabeled_records["text"]:
            # tweet_doc = " ".join(tweet)
            corpus.append(tweet)
        X = tfidf.transform(corpus)
        return classifier.predict(X)


class ML_TEXT_UI(QMainWindow, Ui_main):
    """Main window."""

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_main.__init__(self)
        self.loading("UI", 40)
        self.setupUi(self)
        if colortheme == 2:
            self.setStyleSheet("background: rgba(0,0,0,100%) ")
        self.setWindowIcon(Icon)
        self.setStatusBar(self.statusbar)

        self.initialmenuitems("help", 1)
        self.initialmenuitems("Settings", 1)
        self.loading("toolbar", 90)

        self.status1.setText("﻿Welcome to {}. Press {}+M to see document/help.".format(__programname__, Control_key))
        self.status2.setText('v{}'.format(__version__))
        self.statusbar.addWidget(self.authorLabel)

        self.statusbar.addWidget(self.progressbar)
        self.progressbar.hide()
        self.statusbar.addWidget(self.status1)
        self.statusbar.addPermanentWidget(self.status2)

        self.shortcut0 = QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self)
        self.shortcut0.activated.connect(self.quitfullscreen)

        self.buttonsettings.clicked.connect(self.addSettings)
        self.buttonopen.clicked.connect(self.openfromfile)
        self.buttondata.clicked.connect(self.showdata)
        self.buttonmodel.clicked.connect(self.select_model)
        self.buttonresult.clicked.connect(self.showresult)
        self.buttonfit.clicked.connect(self.fit)
        # self.shortcut5 = QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)
        # self.shortcut5.activated.connect(self.show_fringes)
        self.file = None
        self.data = []
        self.fitting = 0
        self.abortmission = 0
        self.fitobject = None
        self.warningcolor1 = 'red'
        self.warningcolor2 = 'orange'
        self.warningcolor3 = 'royalblue'

        self.models = ["kNN", "SVM", "Neural Network"]
        self.model = "kNN"
        self.para_knn = {"n_neighbors": 5, "metric": 'minkowski', "n_jobs": -1}
        self.para_svm = {"C": 1.0, "kernel": "rbf", "degree": 3, "gamma": "auto", "max_iter": -1, "random_state": None}
        self.num_layers = 2
        self.num_nodes = 4
        self.para_nn = {"hidden_layer_sizes": [self.num_nodes for _ in range(self.num_layers)],
                        "activation": 'relu', "solver": "adam", "alpha": 0.0001,
                        "learning_rate_init": 0.001, "max_iter": 200, "random_state": None}
        self.kernels = ['rbf', 'linear', 'poly', 'sigmoid', 'precomputed']
        self.distance_metrics = ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
        self.solvers = ['adam', 'lbfgs', 'sgd']
        self.activations = ["relu", "identity", "logistic", "tanh"]
        self.trayIcon = None
        self.paras = [self.para_knn, self.para_svm, self.para_nn]
        self.para = self.para_knn

        lb01.hide()
        progressbar.hide()

        self.threadpool = QThreadPool()

    def openfromfile(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        file = dlg.getOpenFileName(self, 'Open file', '', "CSV files (*.CSV *.csv)")

        if file[0] == "":
            return

        self.file = file[0]

        if self.file[-4:None] != ".csv" and self.file[-4:None] != ".CSV":
            self.addlog('{} format is not supported. Please select a .CSV file to open.'.format(self.filename[-4:None]),
                        self.warningcolor2)

    def select_model(self):
        window_load = loadmodel(self, self.models, self.model)
        window_load.show()

        yesorno = window_load.exec_()  # Crucial to capture the result. 1 for Yes and 0 for No.

        result = window_load.result
        if result >= 0:
            self.model = self.models[result]
            self.para = self.paras[result]

    def fit(self):
        if not self.file:
            self.addlog("Please open a data file first.", self.warningcolor2)
            return
        if self.fitting:
            self.abortmission = 1
        else:
            self.addprogressbar()
            self.timer = QTimer()
            self.timer.setInterval(100)
            self.timer.timeout.connect(self.recurring_timer)
            self.timer.start()

            # Pass the function to execute
            worker = Worker(self.execute_fit)  # Any other args, kwargs are passed to the run function
            worker.signals.result.connect(self.process_queue_fit)
            worker.signals.finished.connect(self.thread_complete)
            worker.signals.progress.connect(self.progress)

            # Execute
            self.threadpool.start(worker)

    def execute_fit(self, progress_callback):
        self.fitobject = LoadTEXTDatabase(self.file, self.model, self.para, self, progress_callback)

        if self.abortmission == 1:
            return "ABORT"
        else:
            result = [self.fitobject.score_10FCV, self.fitobject.score_10FCV_std,
                      self.fitobject.score]

        return result

    def process_queue_fit(self, result):

        """Receive result for self.fit()."""

        if result == "ABORT":
            self.addlog("Mission aborted.", "red")
            return

        self.timer.stop()
        self.addlog("Test score: {:.6f}. Total time: {:.2f}s. ".format(result[2], self.totaltime))

    def showdata(self):
        if not self.file:
            self.addlog("Please open a data file first.", self.warningcolor2)
            return
        window_data = data_GUI(self, self.file)
        window_data.show()

    def showresult(self):
        window_result = result_GUI(self)
        window_result.show()

    def Abort_mission(self):
        self.abortmission = 1

    def progress(self, tuple):
        self.progressbar.setValue(tuple[0])
        self.progressbar.setFormat(tuple[1])

    def thread_complete(self):
        self.abortmission = 0
        self.removeprogressbar()
        self.progress((0, "0%"))

    def recurring_timer(self):
        self.totaltime += 0.1

    def addprogressbar(self):
        self.status1.hide()
        self.progressbar.show()
        self.totaltime = 0
        self.buttonfit.setText("ABORT")
        self.buttonfit.setStyleSheet("QPushButton {color : red; }")
        self.fitting = 1

    def removeprogressbar(self):
        self.progressbar.setValue(0)
        self.progressbar.hide()
        self.status1.show()
        self.buttonfit.setText("FIT")
        self.buttonfit.setStyleSheet("QPushButton {color : white; }")
        self.fitting = 0

    def closeEvent(self, event):
        """Make sure the system tray icon is destroyed correctly in Windows. """

        self.trayIcon.hide()
        self.trayIcon.deleteLater()
        event.accept()

    def initialmenuitems(self, item, available):
        if available == 1:
            try:
                getattr(self, "open{}".format(item)).triggered.connect(getattr(self, "add{}".format(item)))
            except AttributeError:
                getattr(self, "open{}".format(item)).setDisabled(True)
        else:
            getattr(self, "open{}".format(item)).setDisabled(True)

    def loading(self, string, percent):
        lb01.setText("Loading {}".format(string))
        progressbar.setValue(percent)
        splash.update()
        app.processEvents()

    def intro_window(self):

        """Show a intro window at the first time launching the program. """

        window_intro = intro_GUI(self)
        window_intro.show()

        yesorno = window_intro.exec_()  # Crucial to capture the result. 1 for Yes and 0 for No.
        if yesorno:
            pass

    def addhelp(self):
        gui = help_GUI(self)
        gui.show()

    def addSettings(self):

        """Customized Settings. """

        window_settings = settings_GUI(self)
        window_settings.show()

        yesorno = window_settings.exec_()  # Crucial to capture the result. 1 for Yes and 0 for No.
        if yesorno:
            pass

    def quitfullscreen(self):
        if self.isFullScreen():
            self.showNormal()

    def addlog(self, string, fg="default", bg="default"):

        """Add a simple text log to the log frame."""

        self.status1.setText(string)
        if fg == "default" and bg == "default":
            pass
        elif fg is not "default" and bg == "default":
            self.status1.setStyleSheet("QLabel {{ color : {}; }}".format(fg))
        elif bg is not "default" and fg == "default":
            self.status1.setStyleSheet("QLabel {{ background-color : {}; }}".format(bg))
        else:
            self.status1.setStyleSheet("QLabel {{ background-color : {}; color : {}; }}".format(bg, fg))


def main():
    global Icon, Control_key, app, splash, lb01, progressbar
    app = QApplication(sys.argv)

    if _platform == "darwin":
        Icon = QIcon(resource_path('Images/icon.icns'))
        Control_key = "⌘"
    else:
        Icon = QIcon(resource_path('Images/icon.ico'))
        Control_key = "Ctrl"

    splash_pix = QPixmap(resource_path('Images/bg.png'))
    splash_pix = splash_pix.scaled(480, 300, transformMode=Qt.SmoothTransformation)
    logo = QPixmap(resource_path('Images/logo_w.png'))
    logo = logo.scaled(320, 120, transformMode=Qt.SmoothTransformation)

    splash = QSplashScreen(splash_pix, Qt.WindowStaysOnTopHint)
    splash.setMask(splash_pix.mask())

    vbox = QVBoxLayout()
    vbox.setContentsMargins(15, 90, 15, 0)
    font = QFont("Helvetica", 15)
    lb00 = QLabel()
    lb00.setAlignment(Qt.AlignCenter)
    lb00.setPixmap(logo)
    lb00.setStyleSheet("background: rgba(0,0,255,0%)")
    lb000 = QLabel("v{}".format(__version__[0:3]))
    lb000.setAlignment(Qt.AlignCenter)
    lb000.setFont(font)
    lb000.setStyleSheet('color: white; background: rgba(0,0,255,0%)')
    lb01 = QLabel("")
    lb01.setAlignment(Qt.AlignCenter)
    lb01.setStyleSheet('color: white; background: rgba(0,0,255,0%)')
    progressbar = QProgressBar()
    progressbar.setTextVisible(False)  # To remove weird text shown next to progressbar in Windows.
    vbox.addWidget(lb00)
    vbox.addStretch()
    vbox.addWidget(lb000)
    vbox.addStretch()
    vbox.addWidget(lb01)
    vbox.addWidget(progressbar)
    splash.setLayout(vbox)

    splash.show()
    app.processEvents()
    window = ML_TEXT_UI()

    window.setWindowTitle("{} v{}".format(__programname__, __version__))
    window.status2.setText('v{}'.format(__version__))
    trayIcon = SystemTrayIcon(Icon, app, window)  # System Tray
    window.trayIcon = trayIcon

    if window_start_view == 1:
        window.showFullScreen()
    elif window_start_view == 2:
        window.showMaximized()
    else:
        window.showNormal()
    if colortheme + darkthemeavailable == 2 or colortheme + darkthemeavailable == 3:
        app.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

    time.sleep(1.2)
    window.show()
    splash.finish(window)

    # Override excepthook to prevent program crashing and create feekback log.
    def excepthook(excType, excValue, tracebackobj):
        """
        Global function to catch unhandled exceptions in main thread ONLY.

        @param excType exception type
        @param excValue exception value
        @param tracebackobj traceback object
        """
        separator = '-' * 80
        logFile = time.strftime("%m_%d_%Y_%H_%M_%S") + ".log"
        notice = \
            """An unhandled exception occurred. \n""" \
            """Please report the problem via email to <%s>.\n""" \
            """A log has been written to "%s".\n\nError information:\n""" % \
            (__emailaddress__, logFile)
        timeString = time.strftime("%m/%d/%Y, %H:%M:%S")

        tbinfofile = io.StringIO()
        traceback.print_tb(tracebackobj, None, tbinfofile)
        tbinfofile.seek(0)
        tbinfo = tbinfofile.read()
        errmsg = '%s: \n%s' % (str(excType), str(excValue))
        sections = [separator, timeString, separator, errmsg, separator, tbinfo]
        msg = '\n'.join(sections)
        try:
            f = open(logFile, "w")
            f.write(msg)
            f.write("Version: {}".format(__version__))
            f.close()
        except IOError:
            pass
        errorbox = QMessageBox()
        errorbox.setText(str(notice) + str(msg) + "Version: " + __version__)
        errorbox.exec_()

    sys.excepthook = excepthook

    if show_intro == 1:
        window.intro_window()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()



