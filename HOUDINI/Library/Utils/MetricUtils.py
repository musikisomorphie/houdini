import os
import warnings
import numpy as np
import pandas as pd
import subprocess
from scipy import stats
import matplotlib.pyplot as plt
import pylatex as pl


class coxsum():
    def __init__(self,
                 index,
                 params,
                 alpha=0.05):
        self.alpha = alpha
        self.ci = 100 * (1 - self.alpha)
        self.z = self._inv_normal_cdf(1 - self.alpha / 2)
        self.index = index
        # be careful with the axis
        self.haz = pd.Series(np.mean(params, axis=0),
                             name='hazard',
                             index=self.index)

        self.se = pd.Series(stats.sem(params, axis=0),
                            name='se',
                            index=self.index)

        self.conf_inv = self._compute_confidence_intervals(self.haz,
                                                           self.se,
                                                           self.ci,
                                                           self.z,
                                                           self.index)

    def _inv_normal_cdf(self, p) -> float:
        return stats.norm.ppf(p)

    def _compute_confidence_intervals(self,
                                      haz,
                                      se,
                                      ci,
                                      z,
                                      index) -> pd.DataFrame:
        # ci = 100 * (1 - self.alpha)
        # z = self._inv_normal_cdf(1 - self.alpha / 2)
        # se = self.standard_errors_
        # hazards = self.params_.values
        conf = pd.DataFrame(np.c_[haz - z * se, haz + z * se],
                            columns=['{}% lower-bound'.format(ci),
                                     '{}% upper-bound'.format(ci)],
                            index=self.index)
        return conf

    # def _compute_standard_errors(self) -> pd.Series:

    #     # if self.robust or self.cluster_col:
    #     #     se = np.sqrt(self._compute_sandwich_estimator(
    #     #         X, T, E, weights).diagonal())
    #     # else:

    #     if hessian_.size > 0:
    #         variance_matrix_ = pd.DataFrame(-inv(hessian_) / np.outer(self._norm_std, self._norm_std),
    #                                         index=X.columns,
    #                                         columns=X.columns)
    #     else:
    #         variance_matrix_ = pd.DataFrame(index=X.columns,
    #                                         columns=X.columns)

    #     se = np.sqrt(variance_matrix_.values.diagonal())
    #     return pd.Series(se, name='se', index=self.params_.index)

    def _compute_z_values(self):
        return self.haz / self.se

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _quiet_log2(self, p):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", r"divide by zero encountered in log2")
            return np.log2(p)

    def summary(self, df_path) -> pd.DataFrame:
        """
        Summary statistics describing the fit.
        Returns
        -------
        df : DataFrame
        """

        filename = str(df_path / 'portec_table')
        with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
            df = pd.DataFrame(index=self.index)
            df.index.name = 'features'
            df['coef'] = self.haz
            df['exp(coef)'] = np.exp(self.haz)
            df['se(coef)'] = self.se
            df['{}% CI(cl)'.format(self.ci)
               ] = self.conf_inv['{}% lower-bound'.format(self.ci)]
            df['{}% CI(cu)'.format(self.ci)
               ] = self.conf_inv['{}% upper-bound'.format(self.ci)]
            df['{}% CI(el)'.format(self.ci)] = np.exp(
                self.haz - self.z * self.se)
            df['{}% CI(eu)'.format(self.ci)] = np.exp(
                self.haz + self.z * self.se)
            df['z'] = self._compute_z_values()
            df['p'] = self._compute_p_values()
            # avoid zero
            df.loc[df['p'] <= 1e-7, 'p'] = 1e-7
            df['-log2(p)'] = -self._quiet_log2(df['p'])
            # 3 digits after decimal
            df.update(df.iloc[:, ].apply(lambda x: (x * 1e2).astype(int) / 1e2))
        doc = pl.Document()
        doc.packages.append(pl.Package('adjustbox'))
        with doc.create(pl.Section('Table')) as Table:
            Table.append(pl.Command('center'))
            Table.append(pl.Command('tiny'))
            Table.append(pl.NoEscape(r'\begin{adjustbox}{width=1\textwidth}'))
            with doc.create(pl.Tabular('c' * (len(df.columns) + 1))) as table:
                table.add_hline()
                table.add_row([df.index.name] + list(df.columns))
                table.add_hline()
                for row in df.index:
                    table.add_row([row] + list(df.loc[row, :]))
                table.add_hline()
            Table.append(pl.NoEscape(r'\end{adjustbox}'))

        doc.generate_pdf(filename, clean_tex=False)
        return df

    def vis_plot(self, scores, plot_path, labels=None, fig_size=4):
        # data = [concord, brier, nbll]
        # brier score: the smaller the better should < 0.25 (concord = 0.5)
        # title = ['Concord Index', 'Brier Score', 'Binomial Log-Likelihood']

        fig, ax = plt.subplots(1,
                               len(scores),
                               figsize=(len(scores) * fig_size, fig_size))
        for ida, axi in enumerate(ax.flat):
            axi.boxplot(scores[ida],
                        vert=True,  # vertical box alignment
                        patch_artist=True,
                        labels=[labels[ida]])  # fill with color
            # if titles is not None:
            #     axi.set_title(titles[ida])
        plt.tight_layout()
        plt.savefig(str(plot_path / 'portec_box_plot.png'))
        plt.figure().clear()
        plt.close()


def main():
    params = np.random.rand(256, 4)
    index = ['age', 'gender', 'grade', 'preOP']
    cox = coxsum(index, params)
    df = cox.summary()
    print(df)


if __name__ == '__main__':
    main()
