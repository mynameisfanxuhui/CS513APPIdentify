import csv
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from general import *

IDLE_TIME = 5  # A flow has been idled larger than this time will be considered as a new action/flow.


class CFlow():

    def __init__(self, sTime):
        self.m_InByteList = []
        self.m_OutByteList = []
        self.m_IntervalList = [0]
        self.m_PacketOutCount = 0
        self.m_PacketInCount = 0
        self.m_PacketOutInRatio = 0
        # self.m_PacketOutMax = 0
        # self.m_PacketOutMin = 0
        # self.m_PacketInMax = 0
        # self.m_PacketInMin = 0

        self.m_ByteOut = 0
        self.m_ByteIn = 0
        self.m_ByteOutInRatio = 0
        self.m_ByteOutMax = 0
        self.m_ByteOutMin = float('inf')
        self.m_ByteInMax = 0
        self.m_ByteInMin = float('inf')
        self.m_ByteInMedian = 0
        self.m_ByteOutMedian = 0
        #variance
        self.m_ByteInVar = 0
        self.m_ByteOutVar = 0


        self.m_StartTime = float(sTime)
        self.m_LastActivityTime = float(sTime)
        self.m_DurationTime = 0
        self.m_IntervalTimeMax = 0
        self.m_IntervalTimeMin = 0
        self.m_IntervalTimeMedian = 0
        self.m_IntervalTimeVariance = 0




        self.m_TimeList = []
        self.m_LenList = []
        self.m_MaxList = []
        self.m_MinList = []
        self.m_MedianList = []
        self.m_VarianceList = []


    def IsActivity(self, sTime):
        if float(sTime) - self.m_StartTime < IDLE_TIME:
            self.m_LastActivityTime = float(sTime)
            fTimeInterval = self.m_LastActivityTime - self.m_IntervalList[-1]
            self.m_IntervalList.append(fTimeInterval)
            return True
        else:
            return False


    def inComePacket(self, bIn, sProtocol, iLen, sInfo):
        if bIn:
            self.m_InByteList.append(iLen)

        else:
            self.m_OutByteList.append(iLen)


    #After flow finished, we need to classify features.
    def FlowFinished(self):
        # time part
        # print("timeInterval list is", self.m_IntervalList)
        self.m_DurationTime = self.m_LastActivityTime - self.m_StartTime
        self.m_IntervalTimeMax = max(self.m_IntervalList)
        self.m_IntervalTimeMin = min(self.m_IntervalList)
        self.m_IntervalTimeMedian = GetMedian(self.m_IntervalList)
        self.m_IntervalTimeVariance = GetVariance(self.m_IntervalList)
        self.m_TimeList.append(self.m_DurationTime)
        self.m_TimeList.append(self.m_IntervalTimeMax)
        self.m_TimeList.append(self.m_IntervalTimeMin)
        self.m_TimeList.append(self.m_IntervalTimeMedian)
        self.m_TimeList.append(self.m_IntervalTimeVariance)






        #TODO: If there is no data, what to do?
        if not self.m_InByteList or not self.m_OutByteList:
            return
        # len part
        self.m_PacketInCount = len(self.m_InByteList)
        # TODO: Currently assume len equal to byte, maybe change later.
        self.m_ByteIn = sum(self.m_InByteList)
        self.m_ByteInMax = max(self.m_InByteList)
        self.m_ByteInMin = min(self.m_InByteList)
        self.m_ByteInMedian = GetMedian(self.m_InByteList)
        self.m_ByteInVar = GetVariance(self.m_InByteList)
        self.m_PacketOutCount = len(self.m_OutByteList)
        self.m_ByteOut = sum(self.m_OutByteList)
        self.m_ByteOutMax = max(self.m_OutByteList)
        self.m_ByteOutMin = min(self.m_OutByteList)
        self.m_ByteOutMedian = GetMedian(self.m_OutByteList)
        self.m_ByteOutVar = GetVariance(self.m_OutByteList)

        # The in cannot be zero because we always init with an in packet.
        self.m_PacketOutInRatio = float(self.m_PacketOutCount) / self.m_PacketInCount
        self.m_ByteOutInRatio = float(self.m_ByteOut) / self.m_ByteIn


        self.m_LenList.append(self.m_PacketOutCount)
        self.m_LenList.append(self.m_PacketInCount)
        self.m_LenList.append(self.m_PacketOutInRatio)
        self.m_LenList.append(self.m_ByteOut)
        self.m_LenList.append(self.m_ByteIn)
        self.m_LenList.append(self.m_ByteOutMin)
        self.m_LenList.append(self.m_ByteOutMax)
        self.m_LenList.append(self.m_ByteOutMedian)
        self.m_LenList.append(self.m_ByteOutInRatio)
        self.m_LenList.append(self.m_ByteInMax)
        self.m_LenList.append(self.m_ByteInMin)
        self.m_LenList.append(self.m_ByteInMedian)
        self.m_LenList.append(self.m_ByteInVar)
        self.m_LenList.append(self.m_ByteOutVar)





        #other part
        self.m_MaxList.append(self.m_ByteOutMax)
        self.m_MaxList.append(self.m_ByteInMax)
        self.m_MaxList.append(self.m_IntervalTimeMax)

        self.m_MinList.append(self.m_ByteOutMin)
        self.m_MinList.append(self.m_ByteInMin)
        self.m_MinList.append(self.m_IntervalTimeMin)

        self.m_MedianList.append(self.m_ByteInMedian)
        self.m_MedianList.append(self.m_ByteOutMedian)
        self.m_MedianList.append(self.m_IntervalTimeMedian)


        self.m_VarianceList.append(self.m_ByteInVar)
        self.m_VarianceList.append(self.m_ByteOutVar)
        self.m_VarianceList.append(self.m_IntervalTimeVariance)





    def ShowAttribute(self):
        print(self.m_PacketOutCount,
              self.m_PacketInCount,
              self.m_PacketOutInRatio,
              self.m_ByteOut,
              self.m_ByteIn,
              self.m_ByteOutInRatio)


    def get_features(self):
        # features = []
        # features.append(self.m_PacketOutCount)
        # features.append(self.m_PacketInCount)
        # features.append(self.m_PacketOutInRatio)
        # features.append(self.m_ByteOut)
        # features.append(self.m_ByteIn)
        # features.append(self.m_ByteOutInRatio)
        # return features
        return self.m_LenList + self.m_TimeList


def GenerateFlow(flie):
    flowList = []  # This will be used for storing flow.
    activeDict = {}  # This is used to store currently active flow.
    with open(flie) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 10000:
                break
            line_count += 1
            sSource = row['Source']
            sDes = row['Destination']
            sProtocol = row['Protocol']
            iLen = int(row['Length'])
            sInfo = row['Info']
            sStartTime = row['Time']

            if (sSource, sDes) not in activeDict and (sDes, sSource) not in activeDict:
                oFlow = CFlow(sStartTime)
                bIn = True
                activeDict[(sSource, sDes)] = oFlow
                oFlow.inComePacket(bIn, sProtocol, iLen, sInfo)
            else:
                if (sSource, sDes) in activeDict:
                    oFlow = activeDict[(sSource, sDes)]
                    bIn = True
                else:
                    oFlow = activeDict[(sDes, sSource)]
                    bIn = False
                # if flow is expired. We need to remove it from activity dict and append it to flow list.
                if not oFlow.IsActivity(sStartTime):
                    oFlow.FlowFinished()
                    flowList.append(oFlow)

                    oNewFlow = CFlow(sStartTime)
                    if bIn:
                        del activeDict[(sSource, sDes)]
                        activeDict[(sSource, sDes)] = oNewFlow
                    else:
                        del activeDict[(sDes, sSource)]
                        activeDict[(sDes, sSource)] = oNewFlow
                if bIn:
                    oFlow = activeDict[(sSource, sDes)]
                else:
                    oFlow = activeDict[(sDes, sSource)]

                oFlow.inComePacket(bIn, sProtocol, iLen, sInfo)

    for oFlow in activeDict.values():
        oFlow.FlowFinished()
        flowList.append(oFlow)

    return flowList


def get_classifier_by_name(classifier_name):
    if classifier_name == "GaussianNB":
        return GaussianNB()
    elif classifier_name == "LogisticRegression":
        return LogisticRegression(solver='lbfgs')
    elif classifier_name == "DecisionTreeClassifier":
        return DecisionTreeClassifier()
    elif classifier_name == "RandomForestClassifier":
        return RandomForestClassifier(n_estimators=50)
    elif classifier_name == "SVM -linear kernel":
        return svm.SVC(kernel='linear')


def get_basic_model_results(X_train, X_test, y_train, y_test):
    scaler = preprocessing.StandardScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    classifiers = [GaussianNB(), LogisticRegression(), DecisionTreeClassifier(),
                   RandomForestClassifier(n_estimators=100),
                   svm.SVC()]
    classifier_names = ["GaussianNB", "LogisticRegression", "DecisionTreeClassifier", "RandomForestClassifier",
                        "SVM -linear kernel"]

    for idx in range(len(classifiers)):
        print("======={}=======".format(classifier_names[idx]))
        train_model(classifier_names[idx], X_train, X_test, y_train, y_test)


def train_model(classifier_name, X_train, X_test, y_train, y_test):
    accuracy_values = []
    precision_values = []
    recall_values = []
    f1_score_values = []

    for i in range(5):
        classifier_clone = get_classifier_by_name(classifier_name)
        classifier_clone.fit(X_train, y_train)

        predicted_output = classifier_clone.predict(X_test)
        accuracy, precision, recall, f1_score_val = get_metrics(y_test, predicted_output, one_hot_rep=False)

        accuracy_values.append(accuracy)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_score_values.append(f1_score_val)

    print_metrics(np.mean(accuracy_values), np.mean(precision_values), np.mean(recall_values), np.mean(f1_score_values))


def print_metrics(accuracy, precision, recall, f1_score_val):
    print("Accuracy : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("Recall : {}".format(recall))
    print("F1 : {}".format(f1_score_val))


def get_metrics(target, logits, one_hot_rep=True):
    if one_hot_rep:
        label = np.argmax(target, axis=1)
        predict = np.argmax(logits, axis=1)
    else:
        label = target
        predict = logits

    accuracy = accuracy_score(label, predict)
    precision = precision_score(label, predict)
    recall = recall_score(label, predict)
    f1_score_val = f1_score(label, predict)

    return accuracy, precision, recall, f1_score_val


def get_train_test_split(samples_features, target_labels):
    X_train, X_test, y_train, y_test = train_test_split(samples_features, target_labels, stratify=target_labels,
                                                        test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    flowList_weibo = GenerateFlow('weibo.csv')
    for oFlow in flowList_weibo:
        print(oFlow.get_features())
    # flowList_douyin = GenerateFlow('douyin.csv')
    # weibo = []
    # douyin = []
    # for oFlow in flowList_weibo:
    #     #         oFlow.ShowAttribute()
    #     weibo.append(oFlow.get_features())
    #
    # for oFlow in flowList_douyin:
    #     douyin.append(oFlow.get_features())
    #
    # weibo = np.array(weibo)
    # douyin = np.array(douyin)
    # feature_array = np.concatenate([weibo, douyin], axis=0)
    # target_labels = np.concatenate([np.ones(len(weibo)), np.zeros(len(douyin))], axis=0)
    # X_train, X_test, y_train, y_test = get_train_test_split(feature_array, target_labels)
    #
    # # show classification result
    # get_basic_model_results(X_train, X_test, y_train, y_test)







