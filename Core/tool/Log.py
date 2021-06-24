#!/usr/bin/env python
# coding:utf-8
import logging
from time import strftime
from time import localtime
import os

class Log():
    def __init__(self, args, num=0):
        regs = eval(args.regs)
        path = args.logpath + "Log/%s_%s/" % (strftime('%Y-%m-%d_%H', localtime()), args.task)
        if not os.path.exists(path):
            os.makedirs(path)
        fpath = path + "%s_embed_size%.4f_lambda1%.7f_reg2%.7f%s" % (
            args.dataset, args.embed_size, regs[0], regs[1], strftime('%Y_%m_%d_%H_%M_%S', localtime()))
        self.logging = logging
        self.logging.basicConfig(filename=fpath,
                                 level=self.logging.INFO)
        print("log to", fpath)
        self.logging.info("begin training %s model ......" % args.model)
        self.logging.info("dataset:%s  embedding_size:%d   dns:%d    batch_size:%d"
                          % (args.dataset, args.embed_size, args.dns, args.batch_size))
        print("dataset:%s  embedding_size:%d   dns:%d   batch_szie:%d" \
              % (args.dataset, args.embed_size, args.dns, args.batch_size))
        if num == 0:
            self.logging.info("regs:%.8f, %.8f  learning_rate:(%.4f, %.4f)"
                         % (regs[0], regs[1], args.lr_embed, args.lr_net))
            print("regs:%.8f, %.8f  learning_rate:(%.4f, %.4f)" \
                  % (regs[0], regs[1], args.lr_embed, args.lr_net))
            print(str(args))
            self.logging.info(str(args))
        else:
            self.logging.info("regs:%.8f, %.8f  learning_rate:%.4f"
                         % (regs[0], regs[1], args.lr))
            print("regs:%.8f, %.8f  learning_rate:%.4f" \
                  % (regs[0], regs[1], args.lr))
            print(str(args))
            self.logging.info(str(args))

    def get_logging(self):
        return self.logging