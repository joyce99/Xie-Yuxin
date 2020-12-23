# -*- coding: utf-8 -*-
import random
import json

def shuffle(lol,seed):
    '''
    lol :: list of list as input
    seed :: seed the shuffling

    shuffle inplace each list in the same order
    '''
    for l in lol:
        random.seed(seed)
        random.shuffle(l)

def contextwin(l, win):
    '''
    win :: int corresponding to the size of the window
    given a list of indexes composing a sentence
    it will return a list of list of indexes corresponding
    to context windows surrounding each word in the sentence
    '''
    assert (win % 2) == 1
    assert win >=1
    l = list(l)

    lpadded = win//2 * [0] + l + win//2 * [0]
    out = [ lpadded[i:i+win] for i in range(len(l)) ]

    assert len(out) == len(l)
    return out

def contextwin_2(ls,win):
    assert (win % 2) == 1
    assert win >=1
    outs=[]
    for l in ls:
        outs.append(contextwin(l,win))
    return outs

def getKeyphraseList(l):
    res, now= [], []
    singleKW = []
    moreKP = []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
                if len(now) == 1:
                    singleKW.append(now[0])
                else:
                    moreKP.append(' '.join(now))
            now = []
    return set(res), set(singleKW), set(moreKP)

def getKeyphraseList_top(l, top_num):
    res, now= [], []
    singleKW = []
    moreKP = []
    for i in range(len(l)):
        if l[i] != 0:
            now.append(str(i))
        if l[i] == 0 or i == len(l) - 1:
            if len(now) != 0:
                res.append(' '.join(now))
                if len(now) == 1:
                    singleKW.append(now[0])
                else:
                    moreKP.append(' '.join(now))
                if len(res) >= top_num:
                    break
            now = []
    return set(res), set(singleKW), set(moreKP)

def conlleval(predictions, groundtruth):
    assert len(predictions) == len(groundtruth)
    res = {}
    all_cnt, good_cnt = len(predictions), 0
    p_cnt, r_cnt = 0, 0
    goodSingleCnt, goodMoreCnt = 0, 0
    singlePreCnt, singleRecallCnt = 0, 0
    morePreCnt, moreRecallCnt = 0, 0
    for i in range(all_cnt):
        # print i
        # if all(predictions[i][0:len(groundtruth[i])] == groundtruth[i]) == True:
        #     good_cnt += 1
        pKeyphraseList, pSingleKpList, pMoreKpList = getKeyphraseList(predictions[i][0:len(groundtruth[i])])
        gKeyphraseList, gSingleKpList, gMoreKpList = getKeyphraseList(groundtruth[i])
        for p in pKeyphraseList:
            for g in gKeyphraseList:
                if p == g:
                    good_cnt += 1
                    break
        for p in pSingleKpList:
            for g in gSingleKpList:
                if p == g:
                    goodSingleCnt += 1
                    break
        for p in pMoreKpList:
            for g in gMoreKpList:
                if p == g:
                    goodMoreCnt += 1
                    break
        p_cnt += len(pKeyphraseList)  #######################
        r_cnt += len(gKeyphraseList)  #######################
        singlePreCnt += len(pSingleKpList)
        singleRecallCnt += len(gSingleKpList)
        morePreCnt += len(pMoreKpList)
        moreRecallCnt += len(gMoreKpList)
        # if len(pKeyphraseList) != 0:
        #     p_cnt += 1
        # if len(gKeyphraseList) != 0:
        #     r_cnt += 1
        # pr_cnt += len(pKeyphraseList & gKeyphraseList)
    res['a'] = 1.0*good_cnt/all_cnt
    res['p'] = 1.0*good_cnt/p_cnt
    res['r'] = 1.0*good_cnt/r_cnt
    res['f'] = 2.0*res['p']*res['r']/(res['p']+res['r'])
    res['SinglePre'] = 1.0 * goodSingleCnt/singlePreCnt
    res['SingleRecall'] = 1.0 * goodSingleCnt/singleRecallCnt
    res['SingleF1'] = 2.0 * res['SinglePre']*res['SingleRecall']/(res['SinglePre']+res['SingleRecall'])
    res['MorePre'] = 1.0 * goodMoreCnt / morePreCnt
    res['MoreRecall'] = 1.0 * goodMoreCnt / moreRecallCnt
    res['MoreF1'] = 2.0 * res['MorePre'] * res['MoreRecall'] / (res['MorePre'] + res['MoreRecall'])
    return res

def conlleval_top(predictions, groundtruth, top_num):
    assert len(predictions) == len(groundtruth)
    res = {}
    all_cnt, good_cnt = len(predictions), 0
    p_cnt, r_cnt = 0, 0
    goodSingleCnt, goodMoreCnt = 0, 0
    singlePreCnt, singleRecallCnt = 0, 0
    morePreCnt, moreRecallCnt = 0, 0
    for i in range(all_cnt):
        pKeyphraseList, pSingleKpList, pMoreKpList = getKeyphraseList_top(predictions[i][0:len(groundtruth[i])], top_num)
        gKeyphraseList, gSingleKpList, gMoreKpList = getKeyphraseList(groundtruth[i])
        for p in pKeyphraseList:
            for g in gKeyphraseList:
                if p == g:
                    good_cnt += 1
                    break
        for p in pSingleKpList:
            for g in gSingleKpList:
                if p == g:
                    goodSingleCnt += 1
                    break
        for p in pMoreKpList:
            for g in gMoreKpList:
                if p == g:
                    goodMoreCnt += 1
                    break
        p_cnt += len(pKeyphraseList)  #######################
        r_cnt += len(gKeyphraseList)  #######################
        singlePreCnt += len(pSingleKpList)
        singleRecallCnt += len(gSingleKpList)
        morePreCnt += len(pMoreKpList)
        moreRecallCnt += len(gMoreKpList)
    res['a'] = 1.0*good_cnt/all_cnt
    res['p'] = 1.0*good_cnt/p_cnt
    res['r'] = 1.0*good_cnt/r_cnt
    res['f'] = 2.0*res['p']*res['r']/(res['p']+res['r'])
    res['SinglePre'] = 1.0 * goodSingleCnt/singlePreCnt
    res['SingleRecall'] = 1.0 * goodSingleCnt/singleRecallCnt
    res['SingleF1'] = 2.0 * res['SinglePre']*res['SingleRecall']/(res['SinglePre']+res['SingleRecall'])
    res['MorePre'] = 1.0 * goodMoreCnt / morePreCnt
    res['MoreRecall'] = 1.0 * goodMoreCnt / moreRecallCnt
    res['MoreF1'] = 2.0 * res['MorePre'] * res['MoreRecall'] / (res['MorePre'] + res['MoreRecall'])
    return res

def countKp20kPhrase():
    validPath = 'data/ACL2017/kp20k/kp20k_train.json'
    testPath = 'data/ACL2017/kp20k/kp20k_test.json'
    validJsonFile = open(validPath, 'r', encoding='utf-8')
    testJsonFile = open(testPath, 'r', encoding='utf-8')
    validSingleNum = 0
    validMoreNum = 0
    validLineNum = 0
    validKpLen = 0.0
    for line in validJsonFile.readlines():
        json_data = json.loads(line)
        for keyword in json_data["keywords"]:
            curKpLen = len(keyword.strip().split(" "))
            validKpLen += curKpLen
            if curKpLen == 1:
                validSingleNum += 1
            elif curKpLen > 1:
                validMoreNum += 1
        validLineNum += 1
    validJsonFile.close()
    testSingleNum = 0
    testMoreNum = 0
    testLineNum = 0
    testKpLen = 0.0
    for line in testJsonFile.readlines():
        json_data = json.loads(line)
        for keyword in json_data["keywords"].split(';'):
            curKpLen = len(keyword.strip().split(" "))
            testKpLen += curKpLen
            if curKpLen == 1:
                testSingleNum += 1
            elif curKpLen > 1:
                testMoreNum += 1
        testLineNum += 1
    testJsonFile.close()
    print("kp20k dataset ----- 训练集single {}个({:.2f}%), more {}个({:.2f}%)；测试集single {}个({:.2f}%), more {}个({:.2f}%)".format(validSingleNum, validSingleNum / (validSingleNum + validMoreNum) * 100.0,
          validMoreNum, validMoreNum / (validSingleNum + validMoreNum) * 100.0, testSingleNum, testSingleNum / (testSingleNum + testMoreNum) * 100.0,
          testMoreNum, testMoreNum / (testSingleNum + testMoreNum) * 100.0))
    print("训练集平均关键短语长度: {:.2f} , 测试集集平均关键短语长度: {:.2f}".format(validKpLen / (validSingleNum + validMoreNum), testKpLen / (testSingleNum + testMoreNum)))

def countKeyphrase(names):
    jsons = []
    for name in names:
        validPath = 'data/ACL2017/' + name + '/' + name + '_valid.json'
        testPath = 'data/ACL2017/' + name + '/' + name + '_test.json'
        validJsonFile = open(validPath, 'r', encoding='utf-8')
        testJsonFile = open(testPath, 'r', encoding='utf-8')
        validSingleNum = 0
        validMoreNum = 0
        validLineNum = 0
        validKpLen = 0.0
        for line in validJsonFile.readlines():
            json_data = json.loads(line)
            for keyword in json_data["keywords"].split(';'):
                curKpLen = len(keyword.strip().split(" "))
                validKpLen += curKpLen
                if curKpLen == 1:
                    validSingleNum += 1
                elif curKpLen > 1:
                    validMoreNum += 1
            validLineNum += 1
        validJsonFile.close()
        testSingleNum = 0
        testMoreNum = 0
        testLineNum = 0
        testKpLen = 0.0
        for line in testJsonFile.readlines():
            json_data = json.loads(line)
            for keyword in json_data["keywords"].split(';'):
                curKpLen = len(keyword.strip().split(" "))
                testKpLen += curKpLen
                if curKpLen == 1:
                    testSingleNum += 1
                elif curKpLen > 1:
                    testMoreNum += 1
            testLineNum += 1
        testJsonFile.close()
        # print("{} dataset ----- 训练集single个数：{}, more 个数：{}；测试集single个数：{}, more 个数：{}".format(name, validSingleNum, validMoreNum, testSingleNum, testMoreNum))
        print("{} dataset ----- 训练集single {}个({:.2f}%), more {}个({:.2f}%)；测试集single {}个({:.2f}%), more {}个({:.2f}%)".format(name,
            validSingleNum, validSingleNum / (validSingleNum + validMoreNum) * 100.0,
            validMoreNum, validMoreNum / (validSingleNum + validMoreNum) * 100.0, testSingleNum,
            testSingleNum / (testSingleNum + testMoreNum) * 100.0,
            testMoreNum, testMoreNum / (testSingleNum + testMoreNum) * 100.0))
        print("训练集平均关键短语长度: {:.2f} , 测试集集平均关键短语长度: {:.2f}".format(validKpLen/(validSingleNum + validMoreNum), testKpLen/(testSingleNum + testMoreNum)))
        print('-------------------------------------------------------------------------------------------------------------------------------')

if __name__ == '__main__':
    names = ['semeval', 'inspec',  'nus', 'krapivin']
    countKp20kPhrase()
    print('-------------------------------------------------------------------------------------------------------------------------------')
    countKeyphrase(names)



