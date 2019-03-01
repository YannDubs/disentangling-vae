import logging
import os


class GraphLogger(object):
    """ Class definition for objects to write data to log files in a
        form which is then easy to be plotted.
    """

    def __init__(self, latent_dims, file_path_name, logger_name):
        """ Create a logger to store information for plotting. """
        self.latent_dim = latent_dims
        self.logger = self.create_logger(logger_name, file_path_name)
        self.set_header()

    def create_logger(self, logger_name, file_path_name):
        """ Create and return a logger object. """
        if os.path.isfile(file_path_name):
            os.remove(file_path_name)

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(file_path_name)
        file_handler.setLevel(logging.DEBUG)
        logger.addHandler(file_handler)
        return logger

    def set_header(self):
        """ Construct and write the header to the log file. """
        header_list = ["Avg-KL-{}".format(i) for i in range(self.latent_dim)]
        header = ",".join(["Epoch"] + header_list)
        self.logger.debug(header)

    def log(self, epoch, kl_div_list):
        """ Write to the log file """
        log_string = ",".join(str(item) for item in [epoch] + kl_div_list)
        self.logger.debug(log_string)
