import os
import warnings
import numpy as np
import pandas as pd
import torch
from pycox.evaluation import EvalSurv
# import torchtuples as tt


class coxph():
    def __init__(self,
                 #  input,
                 pred,
                 target,
                 max_duration=None,
                 sample=None,
                 alpha=0.05):
        self.duration_col = 'duration'
        self.event_col = 'event'
        # self.input = input
        self.pred = pred
        self.target = target
        # alpha for null hypothesis
        self.alpha = alpha

        haz = self._compute_baseline_hazards(max_duration,
                                             sample)
        self.base_haz, self.base_cum_haz = haz

    def _target_to_df(self,
                      target):
        events, durations = target[:, 0], target[:, 1]
        df = pd.DataFrame({self.duration_col: durations,
                           self.event_col: events})
        return df

    def _compute_baseline_hazards(self,
                                  max_duration=None,
                                  sample=None):
        """Computes the Breslow estimates form the data defined by `input` and `target`
        (if `None` use training data).
        Typically call
        model.compute_baseline_hazards() after fitting.

        Keyword Arguments:
            input  -- Input data (train input) (default: {None})
            target  -- Target data (train target) (default: {None})
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            sample {float or int} -- Compute estimates of subsample of data (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            set_hazards {bool} -- Set hazards in model object, or just return hazards. (default: {True})

        Returns:
            pd.Series -- Pandas series with baseline hazards. Index is duration_col.
        """
        if max_duration is None:
            max_duration = np.inf

        df = self._target_to_df(self.target)  # .sort_values(self.duration_col)
        if sample is not None:
            if sample >= 1:
                df = df.sample(n=sample)
            else:
                df = df.sample(frac=sample)

        base_haz = (df.assign(expg=np.exp(self.pred))
                    .groupby(self.duration_col)
                    .agg({'expg': 'sum', self.event_col: 'sum'})
                    .sort_index(ascending=False)
                    .assign(expg=lambda x: x['expg'].cumsum())
                    .pipe(lambda x: x[self.event_col]/x['expg'])
                    .fillna(0.)
                    .iloc[::-1]
                    .loc[lambda x: x.index <= max_duration]
                    .rename('baseline_hazards'))

        assert base_haz.index.is_monotonic_increasing,\
            'The index should be monotonic increasing, as it represents time.'

        base_cum_haz = (base_haz
                        .cumsum()
                        .rename('baseline_cumulative_hazards'))

        return base_haz, base_cum_haz

    # def predict_cumulative_hazards(self,
    #                                input,
    #                                pred,
    #                                max_duration=None):
    #     """See `predict_survival_function`."""
    #     if type(input) is pd.DataFrame:
    #         input = input.values

    #     max_duration = np.inf if max_duration is None else max_duration
    #     bch = self.base_cum_haz
    #     bch = bch.loc[lambda x: x.index <= max_duration]
    #     expg = np.exp(pred).reshape(1, -1)
    #     return pd.DataFrame(bch.values.reshape(-1, 1).dot(expg),
    #                         index=bch.index)

    def predict_surv_df(self,
                        pred,
                        max_duration=None):
        """Predict survival function for `input`. S(x, t) = exp(-H(x, t))
        Require computed baseline hazards.
        Arguments:
            input {np.array, tensor or tuple} -- Input x passed to net.
        Keyword Arguments:
            max_duration {float} -- Don't compute estimates for duration higher (default: {None})
            batch_size {int} -- Batch size (default: {8224})
            baseline_hazards_ {pd.Series} -- Baseline hazards. If `None` used `model.baseline_hazards_` (default: {None})
            eval_ {bool} -- If 'True', use 'eval' mode on net. (default: {True})
            num_workers {int} -- Number of workers in created dataloader (default: {0})
        Returns:
            pd.DataFrame -- Survival estimates. One columns for each individual.
        """

        """See `predict_survival_function`."""
        # if type(input) is pd.DataFrame:
        #     input = input.values

        max_duration = np.inf if max_duration is None else max_duration
        bch = self.base_cum_haz
        bch = bch.loc[lambda x: x.index <= max_duration]
        expg = np.exp(pred).reshape(1, -1)
        surv = np.exp(-pd.DataFrame(bch.values.reshape(-1, 1).dot(expg),
                                    index=bch.index))
        # print(surv.shape)
        return surv

    def eval_surv(self,
                  pred,
                  target):

        surv = self.predict_surv_df(pred)
        events, durations = target[:, 0], target[:, 1]
        ev = EvalSurv(surv, durations, events, censor_surv='km')
        concord = ev.concordance_td()
        time_grid = np.linspace(durations.min(),
                                durations.max(), 100)
        _ = ev.brier_score(time_grid).plot()
        brier = ev.integrated_brier_score(time_grid)
        nbll = ev.integrated_nbll(time_grid)
        return concord, brier, nbll
