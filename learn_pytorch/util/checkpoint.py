import os
import time
import shutil

import torch
import dill


class Checkpoint(object):
    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states'
    MODEL_NAME = 'model.pt'
    INPUT_VOCAB_FILE = 'input_vocab.pt'
    OUTPUT_VOCAB_FILE = 'output_vocab.pt'

    def __init__(self, model, optimizer, epoch, step, input_vocab, output_vocab, path=None):
        self.model = model
        self.optimizer = optimizer
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.epoch = epoch
        self.step = step
        self._path = path

    @property
    def path(self):
        if self._path is None:
            raise LookupError("The checkpoint has not been saved")
        return self._path

    def save(self, experiment_dir):
        date_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self._path = os.path.join(
            experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer}, os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        with open(os.path.join(path, self.INPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.input_vocab, fout)
        with open(os.path.join(path, self.OUTPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.output_vocab, fout)

        return path

    @classmethod
    def load(cls, path):

        if torch.cuda.is_available():
            resume_checkpoint = torch.load(
                os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(
                path, cls.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME),
                               map_location=lambda storage, loc: storage)
        model.flatten_parameters()
        with open(os.path.join(path, cls.INPUT_VOCAB_FILE), 'rb') as f_in:
            input_vocab = dill.load(f_in)
        with open(os.path.join(path, cls.OUTPUT_VOCAB_FILE), 'rb') as f_in:
            output_vocab = dill.load(f_in)
        optimizer = resume_checkpoint['optimizer']
        return Checkpoint(model=model, input_vocab=input_vocab,
                          output_vocab=output_vocab,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          path=path)

    @classmethod
    def get_latest_checkpoint(cls, experiment_path):
        """
        Given the path to an experiment directory, returns the path to the last saved checkpoint's subdirectory.

        Precondition: at least one checkpoint has been made (i.e., latest checkpoint subdirectory exists).
        Args:
            experiment_path (str): path to the experiment directory
        Returns:
             str: path to the last saved checkpoint's subdirectory
        """
        checkpoints_path = os.path.join(
            experiment_path, cls.CHECKPOINT_DIR_NAME)
        all_times = sorted(os.listdir(checkpoints_path), reverse=True)
        return os.path.join(checkpoints_path, all_times[0])
