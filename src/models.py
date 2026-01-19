class Result:
    def __init__(self, value):
        if value == "" or value is None:
            self.value = [None]
            self.technique = [None]
            return
        values = value.split(" (")
        if len(values) > 1:
            number = float(values[0].replace(",", "."))
            self.value = [number if number <= 1 else number / 100]
            self.technique = [values[1].replace(")", "")]
        else:
            number = float(value.replace(",", "."))
            self.value = [number if number <= 1 else number / 100]
            self.technique = [None]

    def add(self, value):
        if value == "" or value is None:
            self.value.append(None)
            self.technique.append(None)
            return
        values = value.split(" (")
        if len(values) > 1:
            number = float(values[0].replace(",", "."))
            self.value.append(number if number <= 1 else number / 100)
            self.technique.append(values[1].replace(")", ""))
        else:
            number = float(value.replace(",", "."))
            self.value.append(number if number <= 1 else number / 100)
            self.technique.append(None)

class Dataset:
    def __init__(self, accuracy, f1, precision, recall, gmean, auc):
        self.accuracy = Result(accuracy)
        self.F1 = Result(f1)
        self.precision = Result(precision)
        self.recall = Result(recall)
        self.gmean = Result(gmean)
        self.auc = Result(auc)

    def add(self, accuracy, f1, precision, recall, gmean, auc):
        self.accuracy.add(accuracy)
        self.F1.add(f1)
        self.precision.add(precision)
        self.recall.add(recall)
        self.gmean.add(gmean)
        self.auc.add(auc)