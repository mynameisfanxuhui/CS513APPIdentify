
import csv

IDLE_TIME = 5 # A flow has been idled larger than this time will be considered as a new action/flow.

class CFlow():

    def __init__(self, sTime):
        self.m_PacketOutCount = 0
        self.m_PacketInCount = 0
        self.m_PacketOutInRatio = 0
        self.m_ByteOut = 0
        self.m_ByteIn = 0
        self.m_ByteOutInRatio = 0
        self.m_StartTime = float(sTime)
        self.m_LastActivityTime = float(sTime)

    def IsActivity(self, sTime):
        self.m_LastActivityTime = float(sTime)
        return self.m_LastActivityTime - self.m_StartTime < IDLE_TIME

    def inComePacket(self, bIn, sProtocol, iLen, sInfo):
        if bIn:
            self.m_PacketInCount += 1
            #TODO: Currently assume len equal to byte, maybe change later.
            self.m_ByteIn += iLen
        else:
            self.m_ByteOut += iLen
            self.m_PacketOutCount += 1
        #The in cannot be zero because we always init with an in packet.
        self.m_PacketOutInRatio = float(self.m_PacketOutCount) / self.m_PacketInCount
        self.m_ByteOutInRatio = float(self.m_ByteOut) / self.m_ByteIn

    def ShowAttribute(self):
        print(self.m_PacketOutCount,
        self.m_PacketInCount,
        self.m_PacketOutInRatio,
        self.m_ByteOut,
        self.m_ByteIn,
        self.m_ByteOutInRatio)




def GenerateFlow():
    flowList = [] # This will be used for storing flow.
    activeDict = {} #This is used to store currently active flow.
    with open('weibo.csv') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count > 100:
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
                    print("not activity")
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
        flowList.append(oFlow)
    return flowList

def Test():
    flowList = GenerateFlow()
    for oFlow in flowList:
        oFlow.ShowAttribute()

Test()


