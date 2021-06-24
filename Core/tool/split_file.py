import os
# path = "/home/zx/Desktop/result/"+"taobao14"+"/"+"conv"+"/"
# path = "/home/zx/Desktop/baseline/"+"taobao14"+"/"+"conv"+"/"
# path = "/home/zx/Desktop/baseline/"+"taobao"+"/"+"auxConvNCF4"+"/2/"
path = "/home/zx/Desktop/draw/"

def file_name(project_path):
    file_list = list()
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if os.path.splitext(file)[1] != '.png':
                file_list.append(os.path.join(root, file))
    return file_list[0]


fr  = open(file_name(path),'r')
line = fr.readline()
file_idx = 1
# fw = open(path + str(file_idx), "w")
while line:
    if "INFO:root:begin training" in line:
        # fw.close()
        fw = open(path + "11"+ str(file_idx), "w")
        file_idx += 1
    fw.write(line)
    line = fr.readline()

fr.close()
fw.close()


j = 0
jigeyizu = 6
for i in range(1, 7):
    fr = open(path + "11" + str(i), "r")
    if (i - 1) % jigeyizu == 0:
        j += 1
        fw = open(path + "0"+str(j), "w")
    line = fr.readline()
    while line:
        fw.write(line)
        line = fr.readline()
    fr.close()
fw.close()

#
j = 0
jigeyizu = 6
for i in range(7, 13):
    fr = open(path + "11" + str(i), "r")
    if (i - 1) % jigeyizu == 0:
        j += 1
        fw = open(path +"00"+ str(j), "w")
    line = fr.readline()
    while line:
        fw.write(line)
        line = fr.readline()
    fr.close()
fw.close()

#
j = 0
jigeyizu = 6
for i in range(13, 19):
    fr = open(path + "11" + str(i), "r")
    if (i - 1) % jigeyizu == 0:
        j += 1
        fw = open(path + "000"+str(j), "w")
    line = fr.readline()
    while line:
        fw.write(line)
        line = fr.readline()
    fr.close()
fw.close()