import matplotlib.pyplot as plt
import pandas as pd


class LogPlotter(object):
    """ Definition of a class for reading in a log file and plotting the results.
    """
    def __init__(self, log_dir, output_dir):
        self.plot(log_dir)

    def plot(self, log_dir):
        """ Display the content of the log file as a line plot"""
        df = pd.read_csv(log_dir)
        headers = list(df)[1:]
        x_axis = list(df)[0]
        df.plot(x=x_axis, y=headers, grid=True, style='-')

        plt.show()
