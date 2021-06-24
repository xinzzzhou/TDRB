# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

class DrawLOG(object):
    def __init__(self, project_path, epoch, loss_epoch,epoch_loss,epoch_hr,epoch_ndcg,epoch_auc,name=""):
        self.markers = ['.', '^', '+', '<', '>', 'x', '*', ',', 'o', 'v', '1', '2', '3', '4', 's', '.']
        self.colors = ['orange', 'g', 'm', 'r', 'tan', 'y', 'brown', 'r', 'linen', 'maroon', 'olive', 'pink', 'black']
        self.epoch = epoch
        self.loss_epoch = loss_epoch
        self.epoch_loss = epoch_loss
        self.epoch_hr = epoch_hr
        self.epoch_ndcg = epoch_ndcg
        self.epoch_auc = epoch_auc
        self.titles = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32]
        # self.titles = [0.0001,0.001,0.01,0.1,1,10,100,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.path = project_path
        self.name=name

    # font = {'weight':'normal',
    #         'size':10
    # }
    def draw(self, pos, x, y, type):
        plt.subplot(2, 2, pos)
        plt.title(type)
        plt.xlabel('epoch')
        plt.ylabel(type)
        for i in range(len(self.loss_epoch)):
            # print("x[i]:"+str(x[i]))
            # print("y[i]:"+str(y[i]))
            # print("marker:" + str(self.markers[i]))
            # print("colors:" + str(self.colors[i]))
            # print("titles:" + str(self.titles[i]))
            plt.plot(x[i], y[i], marker=self.markers[i], color=self.colors[i], label=self.titles[i])
        plt.legend(loc='best')
    def draw_figures(self):
        plt.figure(figsize=(10, 7))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        self.draw(1, self.loss_epoch, self.epoch_loss, "LOSS")
        self.draw(2, self.epoch, self.epoch_hr, "VALIDATE_HR@10")
        self.draw(3, self.epoch, self.epoch_ndcg, "VALIDATE_NDCG@10")
        self.draw(4, self.epoch, self.epoch_auc, "VALIDATE_AUC@10")
        plt.savefig(self.path + "/pic"+self.name+".png")
        plt.close()


def file_name(dataset,model):
    # project_path = "/home/zx/Desktop/draw/"
    project_path = "/home/zx/Desktop/dd/"
    # project_path = "/home/zx/Desktop/baseline/" + dataset+ "/" + model + "/"
    # project_path = "/home/zx/Desktop/result/"+dataset+"/"+model+"/"
    file_list = list()
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if os.path.splitext(file)[1] != '.png':
                file_list.append(os.path.join(root, file))
    return project_path, file_list

def read_file(open_file_path, project_path,name):

    validate_epoch = []
    n = 0
    for i in range(12):
        validate_epoch.append(n)
        n+=50
    for i in range(591,600):
        validate_epoch.append(i)

    # validate_epoch = []
    # n = 0
    # for i in range(12):
    #     validate_epoch.append(n)
    #     n += 50
    # validate_epoch.append(599)
    f = open(open_file_path,'r')
    loss_epoch=[]
    epoch_loss=[]
    epoch = []
    epoch_hr=[]
    epoch_ndcg=[]
    epoch_auc=[]

    tmp_loss_epoch = []
    tmp_epoch_loss = []
    tmp_epoch = []
    tmp_epoch_hr = []
    tmp_epoch_ndcg = []
    tmp_epoch_auc = []

    val_idx = 0

    line = f.readline()
    while line:
        if "INFO:root:begin training " in line:
            if len(tmp_loss_epoch) != 0:
                loss_epoch.append(tmp_loss_epoch)
                epoch_loss.append(tmp_epoch_loss)
                epoch.append(tmp_epoch)
                epoch_hr.append(tmp_epoch_hr)
                epoch_ndcg.append(tmp_epoch_ndcg)
                epoch_auc.append(tmp_epoch_auc)
                tmp_loss_epoch = []
                tmp_epoch_loss = []
                tmp_epoch = []
                tmp_epoch_hr = []
                tmp_epoch_ndcg = []
                tmp_epoch_auc = []
                val_idx = 0
        # loss
        if "] train_loss:" in line:
            a,b = line.split("] train_loss:")
            tmp_loss_epoch.append(int(a.split("INFO:root:[Epoch ")[1]))
            tmp_epoch_loss.append(float(b.split(", ")[0]))
        # validate
        elif " [validate_result]:[" in line:
            _, _, _, hr, ndcg, auc, _, _, _ = line.split(" [validate_result]:[")[1].split("]")[0].split(", ")
            tmp_epoch_hr.append(hr)
            tmp_epoch_ndcg.append(ndcg)
            tmp_epoch_auc.append(auc)
            tmp_epoch.append(validate_epoch[val_idx])
            val_idx += 1
        line = f.readline()
    f.close()


    draw = DrawLOG(project_path, epoch, loss_epoch, epoch_loss, epoch_hr, epoch_ndcg, epoch_auc,name)
    draw.draw_figures()

def draw_log():
    model = "mcbpr"
    project_path, files = file_name("taobao14", model)
    for ff in files:
        read_file(ff, project_path, "")
        # read_file(ff, project_path, ff.split("/")[-1])
if __name__ == '__main__':
    draw_log()