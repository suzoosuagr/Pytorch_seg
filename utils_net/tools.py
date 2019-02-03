import logging

def set_logger(log_path):

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def print_current_losses(epoch, i, losses, iter_time):
    message = '(epoch: {}, iters: {}, t_iter: {})'.format(epoch, i, iter_time)
    for k, v in losses.items():
        message += '[%s] %.3f ' % (k, v)
    logging.info(message)



   

        

     