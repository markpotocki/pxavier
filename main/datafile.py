# loads in the crb or cat file and takes the data we need
import os
import re

class DataFileProcessor:

    _dataDirectory = ""
    _loaded = False


    def setDataDirectory(self, path):
        if self._loaded == True:
            print("unable to set data directory because the processor was already loaded")
            exit(100)
        if os.path.isdir(path) != True:
            print(path + " is not a directory")
            exit(101)
        self._dataDirectory = path


    def loadFiles(self):
        dataFileRegEx = re.compile(r'(\w|\d)*.(hdb)', re.IGNORECASE)
        dataFiles = []
        with os.scandir(self._dataDirectory) as files:
            for f in files:
                if dataFileRegEx.match(f.name):
                    dataFiles.append(f.path)
        self._loaded = True
        return DataFile(dataFiles)


class DataFile:
    _files = []
    _data = []

    def __init__(self, dataList=[]):
        self._files = dataList
        self._getData()

    def _getData(self):
        for fi in self._files:
            with open(fi) as md5File:
                prelimData = md5File.readlines()
                for line in prelimData:
                    foo = line.split(":")
                    self._data.append(foo[0])

    def data(self):
        return self._data


def process(directory):
    dfp = DataFileProcessor()
    dfp.setDataDirectory(directory)
    files = dfp.loadFiles()

    length = len(files.data())
    trainData = []
    validateData = []
    pivotPoint = 0.8 * length
    i = 0

    for f in files.data():
        if i < pivotPoint:
            # training
            trainData.append(f)
        else:
            # validate
            validateData.append(f)
    return trainData, validateData
