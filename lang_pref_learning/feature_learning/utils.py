import os
import datetime
import logging

LANG_MODEL_NAME = {
    # BERT models
    'bert-base': 'google/bert_uncased_L-12_H-768_A-12',
    'bert-mini': 'google/bert_uncased_L-4_H-256_A-4',
    'bert-tiny': 'google/bert_uncased_L-4_H-128_A-2',

    # T5 models
    't5-small': 'google/t5-small',
    't5-base': 'google/t5-base',

}

LANG_OUTPUT_DIM = {
    'bert-base': 768,
    'bert-mini': 256,
    'bert-tiny': 128,

    't5-small': 512,
    't5-base': 768,
}


def timeStamped(fname, fmt='{fname}_%Y%m%d_%H%M%S'):
    """
        Creates a timestamped filename, so we don't override our good work

        Input:
            fname: the given file name
            fmt: the format of timestamp
        Output:
            a new file name with timestamp added
    """
    return datetime.datetime.now().strftime(fmt).format(fname=fname)


def create_logger(exp_dir):
    logger = logging.getLogger("feature_learning")
    logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(stream_formatter)
    logger.addHandler(console_handler)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':.4f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
