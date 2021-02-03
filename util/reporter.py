import numpy as np
from termcolor import cprint


class Reporter:
    def __init__(self, save_root, args):
        self.save_root = save_root
        self.args = args

        self.val_hist = []
        self.test_hist = []
        self.median_hist = []

    def record(self, acc_val, acc_test, acc_median):
        print('')
        cprint(f'Val acc: {acc_val * 100:.2f} %', color='blue', attrs=['bold'])
        cprint(f'Test acc: {acc_test * 100:.2f} %', color='blue', attrs=['bold'])
        cprint(f'Median acc: {acc_median * 100:.2f} %', color='blue', attrs=['bold'])
        print('')

        self.val_hist.append(acc_val)
        self.test_hist.append(acc_test)
        self.median_hist.append(acc_median)

    def report(self):
        with open(self.save_root / 'results.txt', 'w') as file:
            title = f'Results of experiment {self.args.name}: '
            val = f'  - Average val acc in {self.args.iters} runs: ' \
                  f'{np.mean(self.val_hist) * 100:.2f} +/- {np.std(self.val_hist) * 100:.2f} % '
            val_hist = np.array2string(np.array(self.val_hist)*100, separator=', ',
                                       formatter={'float_kind': lambda x: f'{x:.2f}'})

            test = f'  - Average test acc in {self.args.iters} runs: ' \
                   f'{np.mean(self.test_hist) * 100:.2f} +/- {np.std(self.test_hist) * 100:.2f} % '
            test_hist = np.array2string(np.array(self.test_hist)*100, separator=', ',
                                        formatter={'float_kind': lambda x: f'{x:.2f}'})

            median = f'  - Average median acc in {self.args.iters} runs: ' \
                     f'{np.mean(self.median_hist) * 100:.2f} +/- {np.std(self.median_hist) * 100:.2f} % '
            median_hist = np.array2string(np.array(self.median_hist)*100, separator=', ',
                                          formatter={'float_kind': lambda x: f'{x:.2f}'})

            cprint(title, color='blue', attrs=['bold'])
            cprint(val+val_hist, color='blue', attrs=['bold'])
            cprint(test+test_hist, color='blue', attrs=['bold'])
            cprint(median+median_hist, color='blue', attrs=['bold'])

            file.write(title + '\n')
            file.write(val + ', ' + val_hist + '\n')
            file.write(test + ', ' + test_hist + '\n')
            file.write(median + ', ' + median_hist + '\n')