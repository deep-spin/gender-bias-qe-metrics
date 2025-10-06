import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel, wilcoxon
from dataclasses import dataclass
from abc import ABC, abstractmethod
from sklearn.metrics import recall_score, precision_score
from typing import Dict, Union
import pandas as pd
import random

####################
# Continuous Metrics #
####################


class BaseMetric(ABC):
    def __init__(
        self, groups_dict: Dict[str, Union[pd.Series, np.ndarray]], log: bool = False
    ):
        """
        Initializes the metrics evaluation with the given groups.
        Args:
            groups_dict (Dict[str, Union[pd.Series, np.ndarray]]): A dictionary containing two groups 'F' and 'M'.
                - 'F': A pandas Series or numpy array representing the first group.
                - 'M': A pandas Series or numpy array representing the second group.
        Raises:
            AssertionError: If the lengths of the two arrays are not the same.
        """

        x = groups_dict["F"]
        y = groups_dict["M"]

        assert len(x) == len(y), "The two arrays must have the same length."
        assert not np.isnan(x).any(), "Array 'F' contains NaN values."
        assert not np.isnan(y).any(), "Array 'M' contains NaN values."

        self.x = x
        self.y = y
        self.p_tr = 0.05
        self.log = log

    @abstractmethod
    def score(self):
        pass


class Ratio(BaseMetric):
    def score(self):
        """
        Calculate the score by performing a one-sample t-test on the ratio of x to y.
        The method computes the ratio of `self.x` to `self.y`, then performs a one-sample
        t-test to determine if the mean of the ratio is significantly less than 1.
        Returns:
            tuple: A tuple containing:
                - ratio (float): The computed ratio of `self.x` to `self.y`.
                - p_value (float): The p-value from the one-sample t-test.
                - bool: A boolean indicating whether the p-value is less than `self.p_tr`.
        """

        ratio = (100 * self.x) / (100 * self.y)  # Scale to improve numerical stability
        if self.log:
            print("Ratio min", ratio.min())
            print("Ratio max", ratio.max())

        t_stat, p_value = ttest_1samp(ratio, 1, alternative="less")
        return ratio, p_value, p_value < self.p_tr

    def vector(self):
        return (100 * self.x) / (100 * self.y + 1e-9)


class Difference(BaseMetric):
    def score(self):
        """
        Calculate the score by performing a one-sample t-test on the difference
        between two arrays, `self.x` and `self.y`.
        Returns:
            tuple: A tuple containing:
                - diff (array-like): The difference between `self.x` and `self.y`.
                - p_value (float): The p-value from the one-sample t-test.
                - bool: A boolean indicating whether the p-value is less than `self.p_tr`.
        """

        diff = self.x - self.y
        t_stat, p_value = ttest_1samp(diff, 0, alternative="less")
        return diff, p_value, p_value < self.p_tr

    def vector(self):
        return self.x - self.y


class DifferencePairedTTest(BaseMetric):
    def score(self):
        """
        Calculate the score by performing a one-sample t-test on the difference
        between two arrays, `self.x` and `self.y`.
        Returns:
            tuple: A tuple containing:
                - diff (array-like): The difference between `self.x` and `self.y`.
                - p_value (float): The p-value from the one-sample t-test.
                - bool: A boolean indicating whether the p-value is less than `self.p_tr`.
        """

        diff = self.x - self.y
        t_stat, p_value = ttest_rel(self.x, self.y, alternative="less")
        return diff, p_value, p_value < self.p_tr


class DifferenceWilcoxon(BaseMetric):
    def score(self):
        """
        Calculate the score by performing a one-sample t-test on the difference
        between two arrays, `self.x` and `self.y`.
        Returns:
            tuple: A tuple containing:
                - diff (array-like): The difference between `self.x` and `self.y`.
                - p_value (float): The p-value from the one-sample t-test.
                - bool: A boolean indicating whether the p-value is less than `self.p_tr`.
        """

        diff = self.x - self.y
        t_stat, p_value = wilcoxon(self.x, self.y, alternative="less")
        return diff, p_value, p_value < self.p_tr


class DifferenceTTestIND(BaseMetric):
    def score(self):
        """
        Calculate the score by performing a one-sample t-test on the difference
        between two arrays, `self.x` and `self.y`.
        Returns:
            tuple: A tuple containing:
                - diff (array-like): The difference between `self.x` and `self.y`.
                - p_value (float): The p-value from the one-sample t-test.
                - bool: A boolean indicating whether the p-value is less than `self.p_tr`.
        """

        diff = self.x - self.y
        t_stat, p_value = ttest_ind(self.x, self.y, alternative="less")
        return diff, p_value, p_value < self.p_tr


class DifferenceTTestRel(BaseMetric):
    def score(self):
        """
        Calculate the score by performing a one-sample t-test on the difference
        between two arrays, `self.x` and `self.y`.
        Returns:
            tuple: A tuple containing:
                - diff (array-like): The difference between `self.x` and `self.y`.
                - p_value (float): The p-value from the one-sample t-test.
                - bool: A boolean indicating whether the p-value is less than `self.p_tr`.
        """

        diff = self.x - self.y
        t_stat, p_value = ttest_rel(self.x, self.y, alternative="less")
        return diff, p_value, p_value < self.p_tr


def compute_ccdf(values):
    x = np.linspace(0, 1, 10000)
    y = np.array([(values >= i).sum() for i in x], dtype=np.int32)
    y = 100 * y / len(values)
    return x, y


####################
# Discrete Metrics #
####################


class BaseDiscreteMetric(ABC):
    def __init__(self, groups_dict: Dict[str, Union[pd.Series, np.ndarray]]):
        """
        Initialize the evaluation metrics with the given groups.
        Args:
            groups_dict (Dict[str, Union[pd.Series, np.ndarray]]): A dictionary containing the following keys:
                - "F": Feature values for group F.
                - "M": Feature values for group M.
                - "y_true": True labels, which should only contain "male" or "female".
        Raises:
            AssertionError: If `y_true` contains values other than "male" or "female".
            AssertionError: If the lengths of `x`, `y`, and `y_true` are not the same.
        """

        x = groups_dict["F"]
        y = groups_dict["M"]
        y_true = groups_dict["y_true"]

        assert not any(
            y not in ["male", "female"] for y in y_true
        ), "y_true can contain only female or male"

        assert (
            len(x) == len(y) == len(y_true)
        ), "The two arrays must have the same length."

        self.x = x
        self.y = y
        self.y_true = y_true

    def _get_y_pred(self):
        y_pred = np.where(self.x >= self.y, "female", "male")

        # If the predictions are the same for M and F, consider it as an error
        y_pred[self.x == self.y] = np.where(
            self.y_true[self.x == self.y] == "male", "female", "male"
        )
        return y_pred

    @abstractmethod
    def score(self):
        pass


class Accuracy(BaseDiscreteMetric):
    """
    A class used to calculate the accuracy of predictions.
    Methods
    -------
    score():
        Computes the accuracy score by comparing the predicted values to the true values.
    """

    def score(self):
        y_pred = self._get_y_pred()
        return (y_pred == self.y_true).mean()


class GroupMetrics(BaseDiscreteMetric):
    """
    GroupMetrics is a class that evaluates performance metrics for different groups
    (e.g., male and female) in a classification task. It inherits from BaseDiscreteMetric.
    Methods
    -------
    score():
        Computes and returns a dictionary containing recall, precision, and false negative
        rate (FNR) for each group, as well as the ratio of FNRs between the groups.
    """

    def score(self):
        m_true = self.y_true[self.y_true == "male"]
        f_true = self.y_true[self.y_true == "female"]

        y_pred = self._get_y_pred()
        m_pred = y_pred[self.y_true == "male"]
        f_pred = y_pred[self.y_true == "female"]

        recall_f, recall_m = recall_score(
            self.y_true, y_pred, average=None, labels=["female", "male"]
        )

        precision_f, precision_m = precision_score(
            self.y_true, y_pred, average=None, labels=["female", "male"]
        )

        total_error_rate = (y_pred != self.y_true).mean()

        # Error Rates
        fnr_f = (f_pred != f_true).sum() / len(f_true)
        fnr_m = (m_pred != m_true).sum() / len(m_true)

        # Error ratio / Ration between error rates
        er = (fnr_f + 1e-9) / (fnr_m + 1e-9)
        # Error diff  between error rates
        e_diff = fnr_f - fnr_m

        return {
            "total_error_rate": total_error_rate,
            "fnr_ratio": er,
            "fnr_diff": e_diff,
            "female": {
                "recall": recall_f,
                "precision": precision_f,
                "fnr": fnr_f,
            },
            "male": {
                "recall": recall_m,
                "precision": precision_m,
                "fnr": fnr_m,
            },
        }


class GroupMetricsBootstraping(BaseDiscreteMetric):
    """
    GroupMetrics is a class that evaluates performance metrics for different groups
    (e.g., male and female) in a classification task. It inherits from BaseDiscreteMetric.
    Methods
    -------
    score():
        Computes and returns a dictionary containing recall, precision, and false negative
        rate (FNR) for each group, as well as the ratio of FNRs between the groups.
    """

    def __init__(
        self,
        groups_dict: Dict[str, Union[pd.Series, np.ndarray]],
        alpha=0.05,
        n_iter=1000,
        n_samples=100,
        alternative="less",
    ):
        super().__init__(groups_dict)
        self.alpha = alpha  # confidence level for bootstraping
        self.n_iter = n_iter  # num of iterations!
        self.n_samples = n_samples
        self.alternative = alternative
        self.stat_test = wilcoxon

    def score(self):
        m_true = self.y_true[self.y_true == "male"]
        f_true = self.y_true[self.y_true == "female"]

        y_pred = self._get_y_pred()
        m_pred = y_pred[self.y_true == "male"]
        f_pred = y_pred[self.y_true == "female"]

        recall_f, recall_m = recall_score(
            self.y_true, y_pred, average=None, labels=["female", "male"]
        )

        precision_f, precision_m = precision_score(
            self.y_true, y_pred, average=None, labels=["female", "male"]
        )

        total_error_rate = (y_pred != self.y_true).mean()

        # Error Rates
        fnr_f = (f_pred != f_true).sum() / len(f_true)
        fnr_m = (m_pred != m_true).sum() / len(m_true)

        # Error ratio / Ration between error rates
        er = (fnr_f + 1e-9) / (fnr_m + 1e-9)
        # Error diff  between error rates
        e_diff = fnr_f - fnr_m

        return {
            "total_error_rate": total_error_rate,
            "fnr_ratio": er,
            "fnr_diff": e_diff,
            "female": {
                "recall": recall_f,
                "precision": precision_f,
                "fnr": fnr_f,
            },
            "male": {
                "recall": recall_m,
                "precision": precision_m,
                "fnr": fnr_m,
            },
        }

    def stat_significance_with_paired_bootstrap(self):

        original_results = self.score()

        # results_dict = {"total_error_rate":[],"fnr_ratio":[],"fnr_diff":[],"female":{"recall":[],"precision":[],"fnr":[]},"male":{"recall":[],"precision":[],"fnr":[]}}
        # counts_dict = {"total_error_rate":0,"fnr_ratio":0,"fnr_diff":0,"female":{"recall":0,"precision":0,"fnr":0},"male":{"recall":0,"precision":0,"fnr":0}}

        fnr_f = []
        fnr_m = []

        # we will do a paired sampling -- since we want a paired sampled test!
        indexes = [i for i in range(len(self.x))]
        my_x = self.x.copy()
        my_y = self.y.copy()
        my_y_true = self.y_true.copy()
        for i in range(self.n_iter):  # perform n iterations
            s_indices = [
                random.choice(indexes) for _ in range(len(self.x))
            ]  # number of samples must be equal to the number of original data!
            subsampleA = my_x[s_indices]  # paired sampling
            subsampleB = my_y[s_indices]  # paired sampling
            suby_true = my_y_true[s_indices]  # paired sampling

            grouped_metrics = GroupMetrics(
                {"F": subsampleA, "M": subsampleB, "y_true": suby_true}
            ).score()  # calculate metrics
            # results_dict["total_error_rate"].append(grouped_metrics["total_error_rate"])
            # results_dict["fnr_ratio"].append(grouped_metrics["fnr_ratio"])
            # results_dict["fnr_diff"].append(grouped_metrics["fnr_diff"])
            # results_dict["female"]["fnr"].append(grouped_metrics["female"]["fnr"])
            # results_dict["female"]["recall"].append(grouped_metrics["female"]["recall"])
            # results_dict["female"]["precision"].append(grouped_metrics["female"]["precision"])
            # results_dict["male"]["fnr"].append(grouped_metrics["male"]["fnr"])
            # results_dict["male"]["recall"].append(grouped_metrics["male"]["recall"])
            # results_dict["male"]["precision"].append(grouped_metrics["male"]["precision"])
            fnr_f.append(grouped_metrics["female"]["fnr"])
            fnr_m.append(grouped_metrics["male"]["fnr"])

        fnr_f = np.array(fnr_f)
        fnr_m = np.array(fnr_m)
        # Perform stat test
        p_value = self.stat_test(fnr_f, fnr_m, alternative=self.alternative).pvalue
        return {
            "results": original_results,
            "stat_significance": p_value < self.alpha,
            "p_value": p_value,
        }


class GroupMetricsBootstrapingTtest(BaseDiscreteMetric):
    """
    GroupMetrics is a class that evaluates performance metrics for different groups
    (e.g., male and female) in a classification task. It inherits from BaseDiscreteMetric.
    Methods
    -------
    score():
        Computes and returns a dictionary containing recall, precision, and false negative
        rate (FNR) for each group, as well as the ratio of FNRs between the groups.
    """

    def __init__(
        self,
        groups_dict: Dict[str, Union[pd.Series, np.ndarray]],
        alpha=0.05,
        n_iter=1000,
        n_samples=100,
        alternative="less",
    ):
        super().__init__(groups_dict)
        self.alpha = alpha  # confidence level for bootstraping
        self.n_iter = n_iter  # num of iterations!
        self.n_samples = n_samples
        self.alternative = alternative
        self.stat_test = ttest_rel

    def score(self):
        m_true = self.y_true[self.y_true == "male"]
        f_true = self.y_true[self.y_true == "female"]

        y_pred = self._get_y_pred()
        m_pred = y_pred[self.y_true == "male"]
        f_pred = y_pred[self.y_true == "female"]

        recall_f, recall_m = recall_score(
            self.y_true, y_pred, average=None, labels=["female", "male"]
        )

        precision_f, precision_m = precision_score(
            self.y_true, y_pred, average=None, labels=["female", "male"]
        )

        total_error_rate = (y_pred != self.y_true).mean()

        # Error Rates
        fnr_f = (f_pred != f_true).sum() / len(f_true)
        fnr_m = (m_pred != m_true).sum() / len(m_true)

        # Error ratio / Ration between error rates
        er = (fnr_f + 1e-9) / (fnr_m + 1e-9)
        # Error diff  between error rates
        e_diff = fnr_f - fnr_m

        return {
            "total_error_rate": total_error_rate,
            "fnr_ratio": er,
            "fnr_diff": e_diff,
            "female": {
                "recall": recall_f,
                "precision": precision_f,
                "fnr": fnr_f,
            },
            "male": {
                "recall": recall_m,
                "precision": precision_m,
                "fnr": fnr_m,
            },
        }

    def stat_significance_with_paired_bootstrap(self):

        original_results = self.score()

        # results_dict = {"total_error_rate":[],"fnr_ratio":[],"fnr_diff":[],"female":{"recall":[],"precision":[],"fnr":[]},"male":{"recall":[],"precision":[],"fnr":[]}}
        # counts_dict = {"total_error_rate":0,"fnr_ratio":0,"fnr_diff":0,"female":{"recall":0,"precision":0,"fnr":0},"male":{"recall":0,"precision":0,"fnr":0}}

        fnr_f = []
        fnr_m = []

        # we will do a paired sampling -- since we want a paired sampled test!
        indexes = [i for i in range(len(self.x))]
        my_x = self.x.copy()
        my_y = self.y.copy()
        my_y_true = self.y_true.copy()
        for i in range(self.n_iter):  # perform n iterations
            s_indices = [
                random.choice(indexes) for _ in range(len(self.x))
            ]  # number of samples must be equal to the number of original data!
            subsampleA = my_x[s_indices]  # paired sampling
            subsampleB = my_y[s_indices]  # paired sampling
            suby_true = my_y_true[s_indices]  # paired sampling

            grouped_metrics = GroupMetrics(
                {"F": subsampleA, "M": subsampleB, "y_true": suby_true}
            ).score()  # calculate metrics
            # results_dict["total_error_rate"].append(grouped_metrics["total_error_rate"])
            # results_dict["fnr_ratio"].append(grouped_metrics["fnr_ratio"])
            # results_dict["fnr_diff"].append(grouped_metrics["fnr_diff"])
            # results_dict["female"]["fnr"].append(grouped_metrics["female"]["fnr"])
            # results_dict["female"]["recall"].append(grouped_metrics["female"]["recall"])
            # results_dict["female"]["precision"].append(grouped_metrics["female"]["precision"])
            # results_dict["male"]["fnr"].append(grouped_metrics["male"]["fnr"])
            # results_dict["male"]["recall"].append(grouped_metrics["male"]["recall"])
            # results_dict["male"]["precision"].append(grouped_metrics["male"]["precision"])
            fnr_f.append(grouped_metrics["female"]["fnr"])
            fnr_m.append(grouped_metrics["male"]["fnr"])

        fnr_f = np.array(fnr_f)
        fnr_m = np.array(fnr_m)
        # Perform stat test
        p_value = self.stat_test(fnr_f, fnr_m, alternative=self.alternative).pvalue
        return {
            "results": original_results,
            "stat_significance": p_value < self.alpha,
            "p_value": p_value,
        }


####################
# Visualization    #
####################


class CCDF:
    """
    CCDF class for computing Complementary Cumulative Distribution Function (CCDF) metrics.
    Attributes:
        x (list or array-like): Data points for group "F".
        y (list or array-like): Data points for group "M".
    Methods:
        create_points():
            Computes the CCDF points for both groups "F" and "M".
            Returns:
                tuple: x and y coordinates of the CCDF for both groups.
        diff_points():
            Computes the difference in CCDF values between groups "F" and "M".
            Returns:
                tuple: x coordinates and the difference in y coordinates of the CCDF.
        ratio_points():
            Computes the ratio of CCDF values between groups "F" and "M".
            Returns:
                tuple: x coordinates and the ratio of y coordinates of the CCDF.
    """

    def __init__(self, groups_dict):
        x = groups_dict["F"]
        y = groups_dict["M"]
        self.x = x
        self.y = y

    def create_points(self):
        x_ccdf_x, y_ccdf_x = compute_ccdf(self.x)
        x_ccdf_y, y_ccdf_y = compute_ccdf(self.y)
        return x_ccdf_x, y_ccdf_x, x_ccdf_y, y_ccdf_y

    def diff_points(self):
        x_ccdf_x, y_ccdf_x = compute_ccdf(self.x)
        x_ccdf_y, y_ccdf_y = compute_ccdf(self.y)

        y_diff = y_ccdf_x - y_ccdf_y
        return x_ccdf_x, y_diff

    def ratio_points(self):
        x_ccdf_x, y_ccdf_x = compute_ccdf(self.x)
        x_ccdf_y, y_ccdf_y = compute_ccdf(self.y)

        y_ratio = (y_ccdf_x + 1e-9) / (y_ccdf_y + 1e-9)
        return x_ccdf_x, y_ratio
