import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

class LogPlotter(object):
    """ Definition of a class for reading in a log file and plotting the results.
    """
    def __init__(self, log_dir, output_file_name='imgs/avg_kl_per_factor.png', pdf=False):
        """ Initialise log plot """
        plot = self.plot(log_dir)
        print('start plotting yes')
        if output_file_name is None:
            plot.show()
        else:
            if output_file_name[-4:] == '.png' or output_file_name[-4:] == '.pdf':
                plot.savefig(output_file_name)
            else:
                if pdf:
                    plot.savefig(output_file_name + '.pdf')
                else:
                    plot.savefig(output_file_name + '.png')

    def plot(self, log_dir):
        """ Display the content of the log file as a line plot. """
        if not os.path.isfile(log_dir):
            raise Exception('Please ensure that you enter log file path using \'-l\'')
        print('start plotting')
        df = pd.read_csv(log_dir)
        unique_names = df.Loss.unique()
        y = df.loc[df['Loss'] == unique_names[0]].Value
        headers = unique_names
        x_axis = df.Epoch.unique()

        y = np.zeros((len(y),len(unique_names)))
        for i in range(0, len(unique_names)):
            name = unique_names[i]
            y[:,i]=df.loc[df['Loss'] == unique_names[i]].Value
        print(y)
        df.plot(x=x_axis, y=df.loc[df['Loss'] == unique_names[0]].Value, grid=True, style='-')

        return plt
