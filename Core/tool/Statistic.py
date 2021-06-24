def sta(open_file_path, open_file_path1, open_file_path2):
    f = open(open_file_path)
    lens = [0,0,0,0,0,0]
    line = f.readline()
    total_duan = 0
    total_line = 0
    while line:
        total_line += 1
        actions = line.split("|")[1].split(",")
        leng = len(actions)
        if leng == 1:
            lens[0] += 1
        elif leng == 2:
            lens[1] += 1
        elif leng == 3:
            lens[2] += 1
        elif leng == 4:
            lens[3] += 1
        elif leng == 5:
            lens[4]+=1
        else:
            lens[5]+=1
        total_duan += leng
        line = f.readline()
    f.close()
    print("sobazaar-------------")
    print("total_line__%d"%total_line)
    print("total_duan__%d"%total_duan)
    print(lens)
    #
    f = open(open_file_path1)
    lens = [0, 0, 0, 0, 0, 0]
    line = f.readline()
    total_duan = 0
    total_line = 0
    while line:
        total_line += 1
        actions = line.split("|")[1].split(",")
        leng = len(actions)
        if leng == 1:
            lens[0] += 1
        elif leng == 2:
            lens[1] += 1
        elif leng == 3:
            lens[2] += 1
        elif leng == 4:
            lens[3] += 1
        elif leng == 5:
            lens[4] += 1
        else:
            lens[5] += 1
        total_duan += leng
        line = f.readline()
    f.close()
    print("xing-------------")
    print("total_line__%d" % total_line)
    print("total_duan__%d" % total_duan)
    print(lens)
    #
    f = open(open_file_path2)
    lens = [0, 0, 0, 0, 0, 0]
    line = f.readline()
    total_duan = 0
    total_line = 0
    while line:
        total_line += 1
        actions = line.split("|")[1].split(",")
        leng = len(actions)
        if leng == 1:
            lens[0] += 1
        elif leng == 2:
            lens[1] += 1
        elif leng == 3:
            lens[2] += 1
        elif leng == 4:
            lens[3] += 1
        elif leng == 5:
            lens[4] += 1
        else:
            lens[5] += 1
        total_duan += leng
        line = f.readline()
    f.close()
    print("taobao-------------")
    print("total_line__%d" % total_line)
    print("total_duan__%d" % total_duan)
    print(lens)
if __name__ == '__main__':
    path = "/home/zx/Desktop/ConvNCF-master7-2_attention_allduan-linux/Data/sobazaar/sobazaar.train.pos.action"
    path1 = "/home/zx/Desktop/ConvNCF-master7-2_attention_allduan-linux/Data/xing/xing.train.pos.action"
    path2= "/home/zx/Desktop/ConvNCF-master7-2_attention_allduan-linux/Data/taobao/taobao.train.pos.action"
    sta(path, path1, path2)
