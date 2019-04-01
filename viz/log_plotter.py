import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import os

class LogPlotter(object):
    """ Definition of a class for reading in a log file and plotting the results.
    """
    def __init__(self, log_dir, output_file_name=None, pdf=False):
        """ Initialise log plot """
        plot = self.plot(log_dir)

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

        df = pd.read_csv(log_dir)
        headers = list(df)[1:]
        x_axis = list(df)[0]
        df.plot(x=x_axis, y=headers, grid=True, style='-')
        return plt
