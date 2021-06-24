# -*- coding: utf-8 -*-
#!/usr/bin/env python
import os

import matplotlib.pyplot as plt

class Draw(object):
    def __init__(self, args):
        self.markers = ['.', '^', '+', '<', '>', 'x', '*', ',', 'o', 'v', '1', '2', '3', '4', 's', '.']
        self.colors = ['orange', 'g', 'm', 'r', 'tan', 'y', 'brown', 'r', 'linen', 'maroon', 'olive', 'pink',
                  'orange', 'g', 'm', 'r']
        self.args = args
        self.epoch = []
        self.loss_epoch = []
        self.epoch_loss = []
        self.epoch_hr = []
        self.epoch_ndcg = []
        self.epoch_auc = []
        self.titles = []

    # font = {'weight':'normal',
    #         'size':10
    # }
    def draw(self, pos, x, y, type):
        plt.subplot(2, 2, pos)
        plt.title(type)
        plt.xlabel('epoch')
        plt.ylabel(type)
        for i in range(self.args.para_groups):
            plt.plot(x[i], y[i], marker=self.markers[i], color=self.colors[i], label=self.titles[i])
        plt.legend(loc='best')
    def draw_figures(self):
        plt.figure(figsize=(10, 7))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        self.draw(1, self.loss_epoch, self.epoch_loss, "LOSS")
        self.draw(2, self.epoch, self.epoch_hr, "VALIDATE_HR@10")
        self.draw(3, self.epoch, self.epoch_ndcg, "VALIDATE_NDCG@10")
        self.draw(4, self.epoch, self.epoch_auc, "VALIDATE_AUC@10")
        plt.savefig(self.args.logpath + str(self.args.model) + ".png")
        plt.close()

    def draw_brief(self, pos, x, y, type):
        plt.subplot(2, 2, pos)
        plt.title(type)
        plt.xlabel('epoch')
        plt.ylabel(type)
        for i in range(self.args.para_groups):
            plt.plot(x[i], y[i], marker=self.markers[i], color=self.colors[i], label=self.titles[i])
    def draw_figures_brief(self):
        plt.figure(figsize=(10, 7))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        self.draw_brief(1, self.loss_epoch, self.epoch_loss, "LOSS")
        self.draw_brief(2, self.epoch, self.epoch_hr, "VALIDATE_HR@10")
        self.draw_brief(3, self.epoch, self.epoch_ndcg, "VALIDATE_NDCG@10")
        self.draw_brief(4, self.epoch, self.epoch_auc, "VALIDATE_AUC@10")
        plt.savefig(self.args.logpath + str(self.args.model) + "_brief.png")
        plt.close()
    def add_epoch(self, arr):
        self.epoch.append(arr)

    def add_loss_epoch(self, arr):
        self.loss_epoch.append(arr)
    def add_epoch_loss(self, arr):
        self.epoch_loss.append(arr)

    def add_epoch_hr(self, arr):
        self.epoch_hr.append(arr)

    def add_epoch_ndcg(self, arr):
        self.epoch_ndcg.append(arr)

    def add_epoch_auc(self, arr):
        self.epoch_auc.append(arr)

    def add_title(self, arr):
        self.titles.append(arr)


class DrawLOG(object):
    def __init__(self, params, param_nums, project_path, epoch, loss_epoch,epoch_loss,epoch_hr,epoch_ndcg,epoch_auc,name):
        self.markers = ['.', '^', '+', '<', '>', 'x', '*', ',', 'o', 'v', '1', '2', '3', '4', 's', '.']
        self.colors = ['orange', 'g', 'm', 'r', 'tan', 'y', 'brown', 'r', 'linen', 'maroon', 'olive', 'pink',
                  'orange', 'g', 'm', 'r']
        self.epoch = epoch
        self.loss_epoch = loss_epoch
        self.epoch_loss = epoch_loss
        self.epoch_hr = epoch_hr
        self.epoch_ndcg = epoch_ndcg
        self.epoch_auc = epoch_auc
        # self.titles = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.titles = [0.0001,0.001,0.01,0.1,1,10,100,8,9,10,11,12,13,14,15,16,17,18,19,20]
        self.param_nums = param_nums
        self.path = project_path
        self.name=name
        self.params = params

    # font = {'weight':'normal',
    #         'size':10
    # }
    def draw(self, pos, x, y, type):
        plt.subplot(2, 2, pos)
        plt.title(type)
        plt.xlabel('epoch')
        plt.ylabel(type)
        for i in range(self.param_nums):
            plt.plot(x[i], y[i], marker=self.markers[i], color=self.colors[i], label=self.titles[i])
        plt.legend(loc='best')
    def draw_figures(self):
        plt.figure(figsize=(10, 7))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5)
        self.draw(1, self.loss_epoch, self.epoch_loss, "LOSS")
        self.draw(2, self.epoch, self.epoch_hr, "VALIDATE_HR@10")
        self.draw(3, self.epoch, self.epoch_ndcg, "VALIDATE_NDCG@10")
        self.draw(4, self.epoch, self.epoch_auc, "VALIDATE_AUC@10")
        plt.suptitle(self.params)
        plt.savefig(self.path + "/pic"+self.name+".png")
        plt.close()


def file_name(dataset,model,param_dir):
    pwd = os.getcwd()
    # project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..") + "/result/"+dataset+"/"+model+"/com/"+param_dir+"standard/"
    project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..") + "/result/"+dataset+"/"+model+"/com/"+param_dir+"/"
    # project_path = os.path.abspath(os.path.dirname(pwd) + os.path.sep + "..") + "/result/"+dataset+"/"+model+"/"
    file_list = list()
    for root, dirs, files in os.walk(project_path):
        for file in files:
            if os.path.splitext(file)[1] != '.png':
                file_list.append(os.path.join(root, file))
    return project_path, file_list

def read_file(param_nums, open_file_path, project_path, params, name=""):
    f = open(open_file_path)
    epoch = []
    loss_epoch=[]
    epoch_loss=[]
    epoch_hr=[]
    epoch_ndcg=[]
    epoch_auc=[]

    tmp_epoch = []
    tmp_loss_epoch = []
    tmp_epoch_loss = []
    tmp_epoch_hr = []
    tmp_epoch_ndcg = []
    tmp_epoch_auc = []

    n = 0
    for i in range(30):
        tmp_epoch.append(n)
        n+=20
    for i in range(591,600):
        tmp_epoch.append(i)
    for i in range(param_nums):
        epoch.append(tmp_epoch)

    loss_idx = 0
    eva_idx = 0
    line = f.readline()
    while line:
        #loss
        if "] train_loss:" in line:
            tmp_loss_epoch.append(loss_idx)
            tmp_epoch_loss.append(line.split("] train_loss:")[1].split(", ")[0])
            loss_idx += 1
            if loss_idx == 600:
                loss_epoch.append(tmp_loss_epoch)
                epoch_loss.append(tmp_epoch_loss)
                tmp_loss_epoch = []
                tmp_epoch_loss = []
                loss_idx = 0
        #validate
        elif " [validate_result]:[" in line:
            _,_,_,hr,ndcg, auc,_,_,_ = line.split(" [validate_result]:[")[1].split("]")[0].split(", ")
            tmp_epoch_hr.append(hr)
            tmp_epoch_ndcg.append(ndcg)
            tmp_epoch_auc.append(auc)
            eva_idx+=1
            if eva_idx==39:
                epoch_hr.append(tmp_epoch_hr)
                epoch_ndcg.append(tmp_epoch_ndcg)
                epoch_auc.append(tmp_epoch_auc)
                tmp_epoch_hr = []
                tmp_epoch_ndcg = []
                tmp_epoch_auc = []
                eva_idx = 0
        line = f.readline()
    f.close()
    draw = DrawLOG(params, param_nums, project_path, epoch, loss_epoch, epoch_loss, epoch_hr, epoch_ndcg, epoch_auc, name)
    draw.draw_figures()

def draw_log():
    model = "auxConvNCF4"
    choose = 3
    if choose == 1:
        param_nums = 3
        param_dir = 3

        params = ["[0.01,10,1]","[0.01,10,1]","[0.01,10,1]","[0.01,10,1]","[0.01,10,1]"]
        if param_dir == 3:
            params = ["[0.01,10,0.001]","[0.01,10,0.01]","[0.01,10,1]","[0.01,10,10]","[0.01,10,100]"]
        elif param_dir == 2:
            params = ["[0.01,0.001,1]","[0.01,0.01,1]","[0.01,0.1,1]","[0.01,1,1]","[0.01,100,1]"]

        project_path, files = file_name("sobazaar", model, str(param_dir))
        if len(files) == 1:
            read_file(param_nums, files[0], project_path)
        else:
            for f in files:
                f_name = f.split("/")[-1]
                # if f_name == "5":
                #     continue
                print(f_name)
                print(params[int(f_name)-1])
                read_file(param_nums, f, project_path, params[int(f_name)-1], f_name)
    elif choose == 2:
        param_nums = 1
        param_dir = 3
        params = ["[0.01,10,1]", "[0.01,10,1]", "[0.01,10,1]", "[0.01,10,1]", "[0.01,10,1]"]
        if param_dir == 3:
            params = ["[0.01,10,0.001]", "[0.01,10,0.01]", "[0.01,10,1]", "[0.01,10,10]", "[0.01,10,100]"]
        elif param_dir == 2:
            params = ["[0.01,0.001,1]", "[0.01,0.01,1]", "[0.01,0.1,1]", "[0.01,1,1]", "[0.01,100,1]"]
        project_path, files = file_name("sobazaar", model, str(param_dir))
        if len(files) == 1:
            read_file(param_nums, files[0], project_path)
        else:
            for f in files:
                f_name = f.split("/")[-1]
                if f_name != "5":
                    continue
                print(f_name)
                print(params[int(f_name) - 1])
                read_file(param_nums, f, project_path, params[int(f_name) - 1], f_name)
    else:
        param_nums = 2
        param_dir = ""

        project_path, files = file_name("sobazaar", model, str(param_dir))
        if len(files) == 1:
            read_file(param_nums, files[0], project_path)
        else:
            for f in files:
                f_name = f.split("/")[-1]
                read_file(param_nums, f, project_path, "[0.01,10,1]", f_name)

if __name__ == '__main__':
    draw_log()