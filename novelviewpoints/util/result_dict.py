import copy
import os
import pdb
import pickle

import numpy as np
from IPython import embed
from tqdm import tqdm

import torch
from PIL import Image


class ResultDict(object):
    def __init__(
        self, dataset, loggable_metrics, printable_metrics=None, training=False
    ):
        if printable_metrics is None:
            printable_metrics = loggable_metrics

        metric_list = loggable_metrics + printable_metrics
        metric_list = list(set(metric_list))  # uniquifying it

        self.root = dataset.root
        self.data_dict = dataset.data_dict

        self.results = copy.deepcopy(self.data_dict)
        self.labels = copy.deepcopy(self.data_dict)
        self.metrics = copy.deepcopy(self.data_dict)
        self.metric_list = metric_list
        self.printable_metrics = printable_metrics
        self.loggable_metrics = loggable_metrics
        self.summary_stats = dict()
        self.gt = None
        self.training = training

        self.reset(0)

    def reset(self, step):
        self.step = step
        # (c)lass, (m)odel_id, (i)nstance_id
        for cls in self.results:
            self.summary_stats[cls] = {}
            for metric in self.metric_list:
                self.summary_stats[cls][metric] = {
                    "values": [],
                    "stats": dict(),
                }

            for mod in self.results[cls]:
                self.results[cls][mod] = dict()
                self.labels[cls][mod] = dict()
                self.metrics[cls][mod] = dict()

    def update(self, preds, unique_id, metrics=None, labels=None):
        for i in range(len(unique_id)):
            u_id = unique_id[i]
            u_id_split = u_id.split("_")
            cls = u_id_split[0]

            # hack for Pix3D model names with underscores
            m_id = "_".join(u_id_split[1:-1])
            ins_id = u_id_split[-1]

            # id is is str and can be a composition over multiple outputs
            self.results[cls][m_id][ins_id] = dict()
            self.labels[cls][m_id][ins_id] = dict()

            # if not self.training:
            #     for pred in preds:
            #         if "loss" not in pred:
            #             self.results[cls][m_id][ins_id][pred] = (
            #                 preds[pred][i].cpu().numpy()
            #             )
            #             if labels:
            #                 if "viewpoint_distribution" == pred:
            #                     self.labels[cls][m_id][ins_id][pred] = (
            #                         preds[pred][i].cpu().numpy()
            #                     )
            #                 else:
            #                     self.labels[cls][m_id][ins_id][pred] = (
            #                         labels[pred][i].cpu().numpy()
            #                     )

            # add calculated metrics
            inst_metrics = {}
            for _m in metrics:
                inst_metrics[_m] = metrics[_m][i]
            self.metrics[cls][m_id][ins_id] = inst_metrics

    # Calculate Summary Statistics for different  metrics
    def calculate_summary_statistics(self, filter_fn=None):
        # assuming all metrics are calculated for all instances
        for cls in self.metrics:
            for mod in self.metrics[cls]:
                for ins in self.metrics[cls][mod]:
                    instance_results = self.metrics[cls][mod][ins]
                    for metric in self.summary_stats[cls]:
                        self.summary_stats[cls][metric]["values"].append(
                            instance_results[metric]
                        )

            _mask = filter_fn(self.summary_stats[cls]) if filter_fn else None

            # aggregate for cls
            for metric in list(self.summary_stats[cls].keys()):
                if _mask is not None:
                    values = np.array(
                        self.summary_stats[cls][metric]["values"]
                    )[_mask]
                else:
                    values = np.array(self.summary_stats[cls][metric]["values"])
                self.summary_stats[cls][metric]["stats"]["mean"] = np.mean(
                    values
                )
                self.summary_stats[cls][metric]["stats"]["stdev"] = np.std(
                    values
                )
                if metric == "VP_Error":
                    self.summary_stats[cls]["VP_MedError"] = {
                        "stats": {"mean": np.median(values), "stdev": 0}
                    }
                    if "VP_MedError" not in self.printable_metrics:
                        self.printable_metrics.append("VP_MedError")

        def area_under_error_curve(x, max_val, scale=1.0):
            if isinstance(x, list):
                x = np.array(x)
            x = x * scale
            max_val = int(max_val * scale)
            _sum = 0
            for s in range(max_val):
                _sum += (x <= s).mean()
            return _sum / max_val

        # Calculate summary metrics
        SUMMARY_METRICS = {}  # TODO
        for sum_met in SUMMARY_METRICS:
            for cls in self.metrics:
                _HIDDEN_VALUE = None  # figure out if there's a way of doing without in a metric agnostic way
                self.summary_stats[cls][sum_met] = {"stats": {}}
                self.summary_stats[cls][sum_met]["stats"][
                    "mean"
                ] = _HIDDEN_VALUE
                self.summary_stats[cls][sum_met]["stats"]["stdev"] = 0

    def log_performance(self, split, display=True, logger=None):
        step = self.step
        # calculate mean/sdev for class
        _cls = list(self.results.keys())[0]
        _metric = self.printable_metrics[0]

        # class sizes
        class_sizes = np.array(
            [
                len(self.summary_stats[cls][_metric]["values"])
                for cls in self.summary_stats
            ]
        )
        class_sizes = class_sizes / sum(class_sizes)
        dataset_metrics = {}

        for metric in self.summary_stats[_cls]:
            class_means = np.array(
                [
                    self.summary_stats[cls][metric]["stats"]["mean"]
                    for cls in self.summary_stats
                ]
            )
            class_means = (
                class_sizes * class_means
            )  # weighted average by number of instances
            dataset_metrics[metric] = [np.sum(class_means), np.std(class_means)]

        # log if possible
        if logger is not None:
            for metric in self.loggable_metrics:
                logger.add_scalar(
                    "{}/{}".format(metric, split),
                    dataset_metrics[metric][0],
                    step,
                )
                # if metric == 'VP_Error':
                #     _hist = np.array(self.summary_stats[_cls][metric]['values'])
                #     logger.add_histogram("{}/{}".format(metric, split), _hist, step)
        if display:
            self.print_results(dataset_metrics, step)

    def print_results(self, dataset_metrics, step):
        max_metric = [max(len(i), 6) for i in self.printable_metrics]
        metric_printout = [
            "{} {:^%d} |" % max_metric[i] for i in range(len(max_metric))
        ]

        first_line = "{:^12s} |".format("metric")
        dash_line = "{}|".format("-" * 13)
        _cls = list(self.results.keys())[0]
        num_classes = len(self.results)
        for i in range(len(self.printable_metrics)):
            metric = self.printable_metrics[i]
            first_line = metric_printout[i].format(first_line, metric)
            dash_line = "{}{}|".format(dash_line, "-" * (max_metric[i] + 2))

        # print header
        print()
        print("Validation -- Summary Statistics at step {}".format(step))
        print("-" * len(dash_line))
        print(first_line)
        print(dash_line)

        # print per-class performance
        for cls in self.summary_stats:
            print_s = "{:12s} |".format(cls)
            for i in range(len(self.printable_metrics)):
                metric = self.printable_metrics[i]
                # print each performance -- assuming all error < 10000.
                perf_p = "{0[mean]:6^.2f}".format(
                    self.summary_stats[cls][metric]["stats"]
                )
                print_s = metric_printout[i].format(print_s, perf_p)

            print(print_s)

        print_s = "{:12s} |".format("mean")
        for i in range(len(self.printable_metrics)):
            metric = self.printable_metrics[i]
            # print each performance -- assuming all error < 10000.
            multiplier = 100 if "Accuracy" in metric else 1.0
            perf_p = "{:6^.2f}".format(dataset_metrics[metric][0] * multiplier)
            print_s = metric_printout[i].format(print_s, perf_p)

        print(print_s)

    def save(self, path):

        output_dict = {
            "predictions": self.results,
            "metrics": self.metrics,
            "summary_stats": self.summary_stats,
            "labels": self.labels,
        }

        print("Saving output dictionary to {}".format(path))
        with open(path, "wb") as f:
            pickle.dump(output_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
