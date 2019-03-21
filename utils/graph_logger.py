import logging
import os
from utils.helpers import mean


class LossesLogger(object):
    """ Class definition for objects to write data to log files in a
        form which is then easy to be plotted.
    """

    def __init__(self, file_path_name, logger_name='losses_logger', log_level="info"):
        """ Create a logger to store information for plotting. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)

        stream_handler = logging.StreamHandler(__name__)
        stream_handler.setLevel(log_level.upper())

        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(logging.DEBUG)

        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)

        header = ",".join(["Epoch", "Loss", "Value"])
        self.logger.debug(header)

    def log(self, epoch, losses_storer):
        """ Write to the log file """
        for k, v in losses_storer.items():
            log_string = ",".join(str(item) for item in [epoch, k, mean(v)])
            self.logger.debug(log_string)
