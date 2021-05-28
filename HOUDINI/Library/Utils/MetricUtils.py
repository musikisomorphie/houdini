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
                 alpha=0.05,
                 file_nm='portec'):
        self.alpha = alpha
        self.ci = 100 * (1 - self.alpha)
        self.z = self._inv_normal_cdf(1 - self.alpha / 2)
        self.index = index
        self.file_nm = file_nm
        # be careful with the axis
        self.param = pd.Series(np.mean(params, axis=0),
                               name='param',
                               index=self.index)

        self.se = pd.Series(stats.sem(params, axis=0),
                            name='se',
                            index=self.index)

        self.conf_inv = self._compute_confidence_intervals(self.param,
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

        conf = pd.DataFrame(np.c_[haz - z * se, haz + z * se],
                            columns=['{}% lower-bound'.format(ci),
                                     '{}% upper-bound'.format(ci)],
                            index=self.index)
        return conf

    def _compute_z_values(self):
        return self.param / self.se

    def _compute_p_values(self):
        U = self._compute_z_values() ** 2
        return stats.chi2.sf(U, 1)

    def _quiet_log2(self, p):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore',
                                    'divide by zero encountered in log2')
            return np.log2(p)

    def summary(self, df_path) -> pd.DataFrame:
        """
        Summary statistics describing the fit.
        Returns
        -------
        df : DataFrame
        """

        filename = str(df_path / self.file_nm)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore', under='ignore'):
            df = pd.DataFrame(index=self.index)
            df.index.name = 'features'
            df['coef'] = self.param
            df['exp(coef)'] = pd.Series(np.exp(self.param),
                                        name='exp(coef)',
                                        index=self.index)
            df['se(coef)'] = self.se
            df['{}% CI(cl)'.format(self.ci)
               ] = self.conf_inv['{}% lower-bound'.format(self.ci)]
            df['{}% CI(cu)'.format(self.ci)
               ] = self.conf_inv['{}% upper-bound'.format(self.ci)]
            df['{}% CI(el)'.format(self.ci)] = np.exp(
                self.param - self.z * self.se)
            df['{}% CI(eu)'.format(self.ci)] = np.exp(
                self.param + self.z * self.se)
            df['z'] = self._compute_z_values()
            df['p'] = self._compute_p_values()
            # avoid zero
            df.loc[df['p'] <= 1e-16, 'p'] = 1e-16
            df['-log2(p)'] = -self._quiet_log2(df['p'])
            # 4 digits after decimal
            # df.update(df.iloc[:, ].apply(
            #     lambda x: (x * 1e3).astype(int) / 1e3))

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
                        labels=[labels[ida]],
                        showfliers=False)  # fill with color
            # if titles is not None:
            #     axi.set_title(titles[ida])
        plt.tight_layout()
        plt.savefig(str(plot_path / '{}.png'.format(self.file_nm)))
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
