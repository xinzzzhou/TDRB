#!/usr/bin/env python
# coding:utf-8
import math
import numpy as np
from time import time
# from multiprocessing import Pool
# from multiprocessing import cpu_count

np.seterr(divide='ignore',invalid='ignore')

class Evaluator:
    def __init__(self, model, sess, dataset, args, logging):
        self.model = model
        self.sess = sess
        self.dataset = dataset
        self.args = args
        self.logging=logging
        self.eval_feed_dicts = self.init_eval_model()

    def training_loss_acc(self, train_batches):
        train_loss = 0.0
        acc = 0
        num_batch = len(train_batches[1])
        user_input, item_input_pos, item_input_neg = train_batches
        for i in range(len(user_input)):
            feed_dict = {self.model.user_input: user_input[i],
                         self.model.item_input_pos: item_input_pos[i],
                         self.model.item_input_neg: item_input_neg[i]}

            loss, opt_loss, output_pos, output_neg = self.sess.run([self.model.loss, self.model.opt_loss, self.model.output, self.model.output_neg], feed_dict)
            train_loss += loss
            acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
        return train_loss / num_batch, opt_loss/num_batch, acc / num_batch

    def get_loss(self, train_batches):
        train_loss, loss, _ = self.training_loss_acc(train_batches)
        return train_loss, loss
    def output_evaluate(self, train_batches, epoch_count, batch_time, train_time, prev_acc, delta_loss):
        loss_begin = time()
        #train_loss, post_acc = 0,0
        train_loss, loss, post_acc = self.training_loss_acc(train_batches)
        loss_time = time() - loss_begin

        eval_begin = time()
        hr, ndcg, auc, train_auc, validate_result, test_result = self.evaluate()
        eval_time = time() - eval_begin

        res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f train_AUC = %.4f [%.1fs]" \
            " ACC = %.4f train_loss = %.4f delta_loss = %.4f ACC = %.4f [%.1fs]" % ( epoch_count, batch_time, train_time,
                        hr, ndcg, auc, train_auc, eval_time, prev_acc, train_loss, delta_loss, post_acc, loss_time)
        # res = "Epoch %d [%.1fs + %.1fs]: HR = %.4f, NDCG = %.4f AUC = %.4f [%.1fs]" % (epoch_count, batch_time, train_time, hr, ndcg, auc, eval_time)

        self.logging.info(res)
        print(res)

        return post_acc, ndcg, res, hr, auc, validate_result, test_result

    def init_eval_model(self):
        feed_dicts = []
        for u in range(self.dataset.num_users):
            feed_dicts.append(self._evaluate_input(u))
        print("already load the evaluate model...")
        return feed_dicts

    def _evaluate_input(self,user):
        # generate items_list
        item_input = self.dataset.testNegatives[user] # read negative samples from files
        test_item = self.dataset.testRatings[user][1]
        validate_item = self.dataset.validateRatings[user][1]
        item_input.append(test_item)
        item_input.append(validate_item)
        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:,None]
        return user_input, item_input

    def evaluate(self):
        validate_res = []
        test_res = []
        for user in range(self.dataset.num_users):
            validate, test = self._eval_by_user(user)
            validate_res.append(validate)
            test_res.append(test)
        validate_res = np.array(validate_res)
        test_res = np.array(test_res)
        validate_result = (validate_res.mean(axis=0)).tolist()
        test_result = (test_res.mean(axis=0)).tolist()
        # self.logging.info
        # self.logging.info("validate result:"+str(validate_result))
        # self.logging.info("test_result:"+str(test_result))
        hr, ndcg, auc, train_auc = validate_result[3:6] + [0]

        return hr, ndcg, auc, train_auc, validate_result, test_result

    def _eval_by_user(self,user):

        if self.model.train_auc:
            # get predictions of positive samples in training set
            train_item_input = self.dataset.trainList[user]
            train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
            train_item_input = np.array(train_item_input)[:, None]
            feed_dict = {self.model.user_input: train_user_input, self.model.item_input_pos: train_item_input}
            train_predict = self.sess.run(self.model.output, feed_dict)

        # get prredictions of data in testing set
        user_input, item_input = self.eval_feed_dicts[user]
        feed_dict = {self.model.user_input: user_input, self.model.item_input_pos: item_input}

        predictions = self.sess.run(self.model.output, feed_dict)

        neg_predict, test_pos_predict, validate_pos_predict = predictions[:-2], predictions[-2], predictions[-1]
        test_position = (neg_predict >= test_pos_predict).sum()
        validate_position = (neg_predict >= validate_pos_predict).sum()

        validate_ret = []
        validate_ret += self.scoreK(5, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(10, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(20, validate_position, len(predictions)-1)

        test_ret = []
        test_ret += self.scoreK(5, test_position, len(predictions) - 1)
        test_ret += self.scoreK(10, test_position, len(predictions) - 1)
        test_ret += self.scoreK(20, test_position, len(predictions) - 1)

        return validate_ret, test_ret

    def scoreK(self,K, position, negs):
        hr = position < K
        if hr:
            ndcg = math.log(2) / math.log(position + 2)
        else:
            ndcg = 0
        auc = 1 - (position * 1. / negs)  # formula: [#(Xui>Xuj) / #(Items)] = [1 - #(Xui<=Xuj) / #(Items)]
        return hr, ndcg, auc

    def dxyeval(self):
        hr, ndcg, auc, train_auc, validate_result, test_result= self.evaluate()
        res = "Epoch: HR = %.4f, NDCG = %.4f AUC = %.4f train_AUC = %.4f" % (hr, ndcg, auc, train_auc)
        print(res)

class convEvaluator(Evaluator):
    def __init__(self,  model, sess, dataset, args, logging):
        Evaluator.__init__(self, model, sess, dataset, args, logging)
        self.TEST_KEEP_PROB = 1
        self.TRAIN_KEEP_PROB = self.args.keep

    def training_loss_acc(self, train_batches):
        train_loss = 0.0
        acc = 0
        num_batch = len(train_batches[1])
        user_input, item_input_pos, item_input_neg = train_batches
        for i in range(len(user_input)):
            feed_dict = {self.model.user_input: user_input[i],
                         self.model.item_input_pos: item_input_pos[i],
                         self.model.item_input_neg: item_input_neg[i],
                         self.model.keep_prob: self.TEST_KEEP_PROB}

            loss, opt_loss, output_pos, output_neg = self.sess.run([self.model.loss, self.model.opt_loss, self.model.output, self.model.output_neg], feed_dict)
            train_loss += loss
            acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
        return train_loss / num_batch, opt_loss / num_batch, acc / num_batch

    def _eval_by_user(self,user):

        if self.model.train_auc:
            # get predictions of positive samples in training set
            train_item_input = self.dataset.trainList[user]
            train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
            train_item_input = np.array(train_item_input)[:, None]
            feed_dict = {self.model.user_input: train_user_input, self.model.item_input_pos: train_item_input}
            train_predict = self.sess.run(self.model.output, feed_dict)

        # get prredictions of data in testing set
        user_input, item_input = self.eval_feed_dicts[user]
        feed_dict = {self.model.user_input: user_input, self.model.item_input_pos: item_input,
                     self.model.keep_prob: self.TEST_KEEP_PROB}

        predictions = self.sess.run(self.model.output, feed_dict)

        nan_pos = np.argwhere(np.isnan(predictions))
        if len(nan_pos) > 0:
            print("contain nan", nan_pos)
            exit()

        neg_predict, test_pos_predict, validate_pos_predict = predictions[:-2], predictions[-2], predictions[-1]
        test_position = (neg_predict >= test_pos_predict).sum()
        validate_position = (neg_predict >= validate_pos_predict).sum()

        validate_ret = []
        validate_ret += self.scoreK(5, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(10, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(20, validate_position, len(predictions)-1)

        test_ret = []
        test_ret += self.scoreK(5, test_position, len(predictions) - 1)
        test_ret += self.scoreK(10, test_position, len(predictions) - 1)
        test_ret += self.scoreK(20, test_position, len(predictions) - 1)

        return validate_ret, test_ret

class auxEvaluator(convEvaluator):

    def __init__(self, model, sess, dataset, args, logging):
        convEvaluator.__init__(self, model, sess, dataset, args, logging)

    def _allzero(self, arr):
        for a in arr:
            if a != 0:
                return False
        return True

    def all_zero(self,ar):
        ar = ar.split("\t")
        for a in ar:
            a = int(a)
            if a != 0:
                return False
        return True
    def sample_not_zero_duan(self, arr):
        idxs = []
        idx = 0
        for i in arr:
            if not self.all_zero(i):
               idxs.append(idx)
            idx+=1
        if len(idxs) == 0:
            return '0\t0\t0\t0'
        # if len(idxs) == 1:
        #     return arr[idxs[0]]
        return arr[np.random.randint(len(idxs))]

    def splitAsample(self, strr):
        arr = strr.split(",")
        duan = len(arr)
        if duan == 1:
            aaa = arr[0].split("\t")
        else:
            aaa = self.sample_not_zero_duan(arr).split("\t")
        aaa = np.array(aaa, dtype=np.int32)
        aaa = aaa.astype(np.int32)
        return aaa

    def splitAsample_train(self, arr):
        duan = len(arr)
        if duan == 1:
            aaa = arr[0].split("\t")
        else:
            aaa =  self.sample_not_zero_duan(arr).split("\t")
        aaa = np.array(aaa, dtype=np.int32)
        aaa = aaa.astype(np.int32)
        return aaa

    def training_loss_acc(self, train_batches):
        train_loss = 0.0
        acc = 0
        num_batch = len(train_batches[1])
        user_input, item_input_pos, item_input_neg, action_input_pos, action_input_neg = train_batches
        for i in range(len(user_input)):
            feed_dict = {self.model.user_input: user_input[i],
                         self.model.item_input_pos: item_input_pos[i],
                         self.model.item_input_neg: item_input_neg[i],
                         self.model.action_input_pos: action_input_pos[i],
                         self.model.action_input_neg: action_input_neg[i],
                         self.model.keep_prob: self.TEST_KEEP_PROB}

            loss, opt_loss, output_pos, output_neg = self.sess.run([self.model.loss, self.model.opt_loss, self.model.output, self.model.output_neg], feed_dict)
            train_loss += loss
            acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
        return train_loss / num_batch, opt_loss / num_batch, acc / num_batch

    def _evaluate_input(self, user):
        # generate items_list
        item_input = self.dataset.testNegatives[user]  # read negative samples from files
        # negative-action
        action_input = np.zeros([len(item_input) + 2, self.args.len_sequence])
        action_gate = np.zeros([len(item_input) + 2, 1])  # default don't have action
        idx = 0
        for i in item_input:
            key = str(user) + "-" + str(i)
            if self.dataset.trainAuxiliaryNegActionMap.get(key) != None:
                action_gate[idx] = np.full([1, 1], self.args.gate)
                aaa = self.dataset.trainAuxiliaryNegActionMap.get(key)
                aaa = self.splitAsample_train(aaa)
                action_input[idx, :] = aaa
            idx = idx + 1
        test_item = self.dataset.testRatings[user][1]

        item_input.append(test_item)
        test = self.dataset.testAuxiliaryActionList[user]
        tmp_test = self.splitAsample(test)
        # tmp_test = list(map(int, tmp_test))
        action_input[-2, :] = tmp_test
        # print("u:" + str(user) + "_i:" + str(test_item) + "_t:" + str(test))
        if not self._allzero(tmp_test):
            action_gate[-2] = np.full([1, 1], self.args.gate)
        # validate
        validate_item = self.dataset.validateRatings[user][1]
        item_input.append(validate_item)
        validate = self.dataset.validateAuxiliaryActionList[user]
        # print("u:"+str(user)+"_i:"+str(validate_item) +"_v:"+ str(validate))
        tmp_validate = self.splitAsample(validate)
        action_input[-1, :] = tmp_validate
        if not self._allzero(tmp_validate):
            action_gate[-1] = np.full([1, 1], self.args.gate)
        #
        user_input = np.full(len(item_input), user, dtype='int32')[:, None]
        item_input = np.array(item_input)[:, None]
        return user_input, item_input, action_input, action_gate

    def _eval_by_user(self,user):

        if self.model.train_auc:
            # get predictions of positive samples in training set
            train_item_input = self.dataset.trainList[user]
            train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
            train_item_input = np.array(train_item_input)[:, None]
            feed_dict = {self.model.user_input: train_user_input, self.model.item_input_pos: train_item_input}
            train_predict = self.sess.run(self.model.output, feed_dict)

        # get prredictions of data in testing set
        user_input, item_input, action_input, action_gate = self.eval_feed_dicts[user]
        feed_dict = {self.model.user_input: user_input, self.model.item_input_pos: item_input,
                     self.model.action_input_pos: action_input,
                     self.model.keep_prob: self.TEST_KEEP_PROB}

        predictions = self.sess.run(self.model.output, feed_dict)

        nan_pos = np.argwhere(np.isnan(predictions))
        if len(nan_pos) > 0:
            print("contain nan", nan_pos)
            exit()

        neg_predict, test_pos_predict, validate_pos_predict = predictions[:-2], predictions[-2], predictions[-1]
        test_position = (neg_predict >= test_pos_predict).sum()
        validate_position = (neg_predict >= validate_pos_predict).sum()

        validate_ret = []
        validate_ret += self.scoreK(5, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(10, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(20, validate_position, len(predictions)-1)

        test_ret = []
        test_ret += self.scoreK(5, test_position, len(predictions) - 1)
        test_ret += self.scoreK(10, test_position, len(predictions) - 1)
        test_ret += self.scoreK(20, test_position, len(predictions) - 1)

        return validate_ret, test_ret

class auxGateEvaluator(auxEvaluator):
    def __init__(self, model, sess, dataset, args, logging):
        auxEvaluator.__init__(self, model, sess, dataset, args, logging)

    def training_loss_acc(self, train_batches):
        train_loss = 0.0
        acc = 0
        num_batch = len(train_batches[1])
        user_input, item_input_pos, item_input_neg, action_input_pos, action_input_neg, action_gate_pos, action_gate_neg = train_batches
        for i in range(len(user_input)):
            feed_dict = {self.model.user_input: user_input[i],
                         self.model.item_input_pos: item_input_pos[i],
                         self.model.item_input_neg: item_input_neg[i],
                         self.model.action_input_pos: action_input_pos[i],
                         self.model.action_input_neg: action_input_neg[i],
                         self.model.action_gate_pos: action_gate_pos[i],
                         self.model.action_gate_neg: action_gate_neg[i],
                         self.model.keep_prob: self.TEST_KEEP_PROB}

            loss, opt_loss, output_pos, output_neg = self.sess.run([self.model.loss, self.model.opt_loss, self.model.output, self.model.output_neg], feed_dict)
            train_loss += loss
            # if train_loss ==
            acc += ((output_pos - output_neg) > 0).sum() / len(output_pos)
        return train_loss / num_batch, opt_loss / num_batch, acc / num_batch

    def _eval_by_user(self,user):

        if self.model.train_auc:
            # get predictions of positive samples in training set
            train_item_input = self.dataset.trainList[user]
            train_user_input = np.full(len(train_item_input), user, dtype='int32')[:, None]
            train_item_input = np.array(train_item_input)[:, None]
            feed_dict = {self.model.user_input: train_user_input, self.model.item_input_pos: train_item_input}
            train_predict = self.sess.run(self.model.output, feed_dict)

        # get prredictions of data in testing set
        user_input, item_input, action_input, action_gate = self.eval_feed_dicts[user]
        feed_dict = {self.model.user_input: user_input, self.model.item_input_pos: item_input,
                     self.model.action_input_pos: action_input, self.model.action_gate_pos:action_gate,
                     self.model.keep_prob: self.TEST_KEEP_PROB}

        predictions = self.sess.run(self.model.output, feed_dict)

        nan_pos = np.argwhere(np.isnan(predictions))
        if len(nan_pos) > 0:
            print("contain nan", nan_pos)
            exit()

        neg_predict, test_pos_predict, validate_pos_predict = predictions[:-2], predictions[-2], predictions[-1]
        test_position = (neg_predict >= test_pos_predict).sum()
        validate_position = (neg_predict >= validate_pos_predict).sum()

        validate_ret = []
        validate_ret += self.scoreK(5, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(10, validate_position, len(predictions)-1)
        validate_ret += self.scoreK(20, validate_position, len(predictions)-1)

        test_ret = []
        test_ret += self.scoreK(5, test_position, len(predictions) - 1)
        test_ret += self.scoreK(10, test_position, len(predictions) - 1)
        test_ret += self.scoreK(20, test_position, len(predictions) - 1)

        return validate_ret, test_ret