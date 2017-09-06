import chainer
from chainer import Variable
import numpy as np
import math


class Evaluator(object):
    def __init__(self, gpu, n_class):
        self.gpu = gpu
        self.n_class = n_class

    def preparation(self, predictions, truths):
        if self.gpu:
            gpu_predictions = predictions.data
            gpu_truths = truths.data

            cpu_predictions = chainer.cuda.to_cpu(gpu_predictions)
            cpu_truths = chainer.cuda.to_cpu(gpu_truths)
        else:
            cpu_predictions = predictions.data
            cpu_truths = truths.data

        # we want to exclude labels with -1
        mask = cpu_truths != -1
        # reduce values along classe axis
        reduced_cpu_preditions = np.argmax(cpu_predictions, axis=1)

        # mask
        self.masked_reduced_cpu_preditions = reduced_cpu_preditions[mask]
        self.masked_cpu_truths = cpu_truths[mask]
        self.compute_evaluation_values()

    def calculate_confusion_matrix_values(self, i_class):
        tmp_pred = self.masked_reduced_cpu_preditions.copy()
        tmp_trth = self.masked_cpu_truths.copy()
        tmp_pred[tmp_pred!=i_class] = -3
        tmp_trth[tmp_trth!=i_class] = -2
        gt_pixel_size = (tmp_trth!=-2).sum()
        pred_pixel_size = (tmp_pred!=-3).sum()

        true_positive = (tmp_pred==tmp_trth).sum()
        false_positive = pred_pixel_size - true_positive
<<<<<<< HEAD
        tp_fp_fn = gt_pixel_size+pred_pixel_size-true_positive
        return gt_pixel_size, true_positive, false_positive, tp_fp_fn


=======
        tp_fp_fn = gt_pixel_size + pred_pixel_size - true_positive
        print(gt_pixel_size, true_positive, false_positive, tp_fp_fn)
        return gt_pixel_size, true_positive, false_positive, tp_fp_fn

>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
    def compute_evaluation_values(self):
        self.iou_result = []
        self.ar_result = []
        self.ap_result = []
        self.roc_result = []
        for i_class in range(self.n_class):
            gt_pixel_size, true_positive, false_positive, tp_fp_fn = \
                self.calculate_confusion_matrix_values(i_class)
            if gt_pixel_size==0:
                self.iou_result.append(0)
                self.ar_result.append(0)
                self.ap_result.append(0)
            else:
<<<<<<< HEAD
                self.iou_result.append(true_positive.sum()/tp_fp_fn)
                self.ar_result.append(true_positive/gt_pixel_size)
                self.ap_result.append(true_positive/(true_positive+false_positive))

            for thresh in np.arange(0, 1.1, 0.1):
                if gt_pixel_size==0:
                    self.roc_result.append([i_class, thresh, 0, 0, 0])  # tp, fp, num of sumple
                else:
                    tp = 1 if float(true_positive)/gt_pixel_size>=thresh else 0
                    fp = 1 if float(false_positive)/gt_pixel_size>=thresh else 0
                    self.roc_result.append([i_class, thresh, tp, fp, 1])  # tp, fp, num of sumple
        self.iou_result = np.array(self.iou_result)
        self.ar_result = np.array(self.ar_result)
        self.ap_result = np.array(self.ap_result)
        self.roc_result = np.array(self.roc_result)
=======
                self.iou_result.append(true_positive/tp_fp_fn)
                self.ar_result.append(true_positive/gt_pixel_size)
                self.ap_result.append(true_positive/(true_positive+false_positive))

        self.iou_result = np.array(self.iou_result)
        self.ar_result = np.array(self.ar_result)
        self.ap_result = np.array(self.ap_result)
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946

    def get_accuracy(self):
        return (self.masked_reduced_cpu_preditions == self.masked_cpu_truths).mean()

    def get_iou(self):
        return self.iou_result

    def get_ar(self):
        return self.ar_result

    def get_ap(self):
        return self.ap_result
<<<<<<< HEAD

    def get_roc_curve_values(self):
        return self.roc_result
=======
>>>>>>> 23dd86731ea147188fd490755d5fb2872295a946
