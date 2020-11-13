import statistics





def GetMedian(targetList):
    return sorted(targetList)[len(targetList) // 2]

def GetVariance(targetList):
    if len(targetList) < 2:
        return 0
    return statistics.variance(targetList)

