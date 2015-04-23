# coding:utf8


def getNextids(targetid, idlist, tokenlist, taglist, headlist, visited, targetName, direction=None):
    index = idlist.index(targetid)  # index表示 第几行
    next_targetids = []  # 接下来要遍历的点
    result = []
    
    # 方向向向右的点
    if headlist[index] != 0 and headlist[index] not in visited:  
        result.append((targetName, taglist[index], direction == None and '->' or direction, tokenlist[idlist.index(headlist[index])]))
        next_targetids.append((headlist[index], direction == None and '->' or direction))
        visited.add(headlist[index])

    # 方向向左的点
    for i, hid in enumerate(headlist):
        if hid == targetid and idlist[i] not in visited:
            result.append((targetName, taglist[i], direction == None and '<-' or direction, tokenlist[i]))
            next_targetids.append((idlist[i], direction == None and '<-' or direction))
            visited.add(idlist[i])
    return next_targetids, result  
    
def BFSrecurse(targetid, idlist, tokenlist, taglist, headlist, targetName, direction=None, depth=5):
#     print 'target', tokenlist[idlist.index(targetid)]
    result = []
    visited = set()
    visited.add(targetid)
    
    next_targetids, r = getNextids(targetid, idlist, tokenlist, taglist, headlist, visited, targetName, direction)
    result = result + r
    
    while depth > 0 :
        new_next_targetids = []
        for nextid, direction in next_targetids:
            new_next_targetids_thisnode, r = getNextids(nextid, idlist, tokenlist, taglist, headlist, visited, targetName, 'ex')
            new_next_targetids = new_next_targetids + new_next_targetids_thisnode
            result = result + r
#         print [tokenlist[idlist.index(node)] for node in visited]
        next_targetids = new_next_targetids
        depth -= 1
#     print result
    return result
    

# 处理一句话，把这句话变成tuple，即links
def handleBlock(block, treewindow=5, contextwindow=5):
    tuples = []
    tokenlist = []
    headlist = []
    taglist = []
    idlist = []
    for line in block:
        splits = line.split('\t')
        idlist.append(int(splits[0]))
        tokenlist.append(splits[1].strip())
        headlist.append(int(splits[6]))
        taglist.append(splits[7].strip())
    for i, target in enumerate(tokenlist):
        for j in range(max(0, i - contextwindow), min(i + contextwindow + 1, len(tokenlist))):
            if j != i:
                tuples.append((tokenlist[i], 'surroundword', '-', tokenlist[j]))
        tuples = tuples + BFSrecurse(idlist[i], idlist, tokenlist, taglist, headlist, target, direction=None, depth=5)
    return tuples
      
      
# # 处理一句话，把这句话变成tuple，即links
# def handleBlock(block, treewindow=5, contextwindow=5):
#     tuples = []
#     tokenlist = []
#     headlist = []
#     taglist = []
#     idlist = []
#     for line in block:
#         splits = line.split('\t')
#         idlist.append(int(splits[0]))
#         tokenlist.append(splits[1])
#         headlist.append(int(splits[6]))
#         taglist.append(splits[7])
#     for i, target in enumerate(tokenlist):
#         for j in range(max(1, i - contextwindow), min(i + contextwindow, len(tokenlist))):
#             if j != i:
#                 tuples.append((tokenlist[i].lower(), 'surroundword', '-', tokenlist[j].lower()))
#         tuples = tuples + BFSrecurse(idlist[i], idlist, tokenlist, taglist, headlist, target, direction=None, depth=5)
#     return tuples
#         
def getReverseData(data):
    reverseData = {}
    for k in data:
        v = data[k]
        target, tag, direction = k
        if direction == '->':
            direction = '<-'
        elif direction == '<-':
            direction = '->'
        for word in v:
            reverseK = (word, tag, direction)
            if reverseK in reverseData:
                reverseData[reverseK].append(target)
            else:
                reverseData[reverseK] = [target]
    return reverseData

def writeData(data, writer, maxSentLength, merge=False):
    for k in data:
        v = data[k]
        mergePart = ''
        if merge:
            mergePart = k[1] + '_' + k[2] + ' None'
        else:
            mergePart = k[1] + ' ' + k[2]
        
            
        if len(v) > maxSentLength:
            batch = len(v) / maxSentLength + 1
            for i in range(batch):
                linesize = '0'
                if i == batch - 1:
                    linesize = str(len(v) - (batch - 1) * maxSentLength)
                else:
                    linesize = str(maxSentLength)
                line = k[0] + ' ' + linesize + ' ' + mergePart + ' 1 ' + ' '.join(v[maxSentLength * i: min(maxSentLength * (i + 1), len(v))]) 
                line = line.strip()
                writer.write(line + '\n')
        else:
            line = k[0] + ' ' + str(len(v)) + ' ' + mergePart + ' 1 ' + ' '.join(v) 
            line = line.strip()
            writer.write(line + '\n')


# 把gdep文件转换成graph2vec的格式
# target relationcount relationtype relationdirection weight word0 word1 word2...

# 应该考虑逆向link，接下来要试验一下
def transGDEP2graphFormat(gdepFiles, resultFile, withReverse=False, merge=False):
    flushSize = 2000000  # 每隔这么多句处理完之后就写到文件里，否则一直堆在内存里会导致内存消耗殆尽
    maxSentLength = 500  # 一行中words个数的上限，防止每行有过多的单词，保证graph2vec的数组不会越界
    writer = open(resultFile, 'w')
    data = {}
    block = []
    count = 0
    for gdepFile in gdepFiles:
        for line in open(gdepFile):              
            line = line.strip()
            if line.strip() == '':
                count += 1
                if count % 100000 == 0:
                    print count, "sentences read"
                if count % flushSize == 0:
                    writeData(data, writer, maxSentLength, merge=merge)
                    if withReverse:
                        reverseData = getReverseData(data)
                        writeData(reverseData, writer, maxSentLength, 'in', merge)
                        reverseData = {}  
                    data = {}
                    print str(count), 'lines wrote'
                    print '----------------------'
                blockData = handleBlock(block)
                for target, tag, direction, head in blockData:
                    k = (target, tag, direction) 
                    if k in data:
                        data[k].append(head)
                    else:
                        data[k] = [head]
                block = []
#                 break
            else:
                block.append(line)
    if len(data) != 0:
        writeData(data, writer, maxSentLength, merge)
        if withReverse:
            reverseData = getReverseData(data)
            writeData(reverseData, writer, maxSentLength, merge)
        print str(count), 'lines wrote'
        print '----------------------'
    writer.close()


# 应该考虑逆向link，接下来要试验一下
def transGDEP2graphFormat_v2(gdepFiles, resultFile, withReverse=False, merge=False):
    writer = open(resultFile, 'w')
    block = []
    count = 0
    for gdepFile in gdepFiles:
        for line in open(gdepFile):              
            line = line.strip()
            if line.strip() == '':
                count += 1
                blockData = handleBlock(block)
                for rel in  blockData:
                    writer.write(' '.join(rel))
                    writer.write('\n')
                block = []
            else:
                block.append(line)
    if block != []:
        for rel in handleBlock(block):
            writer.write(' '.join(rel))
            writer.write('\n')
    writer.close()




 

