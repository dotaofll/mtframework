import logging

import torchtext


class SourceField(torchtext.data.Field):
    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be True")
        kwargs['batch_first'] = True

        if kwargs.get('include_lengths') is False:
            logger.warning(
                "Option include_lengths has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['include_lengths'] = True


        super(SourceField, self).__init__(**kwargs)


class TargetField(torchtext.data.Field):

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_size has to be True")
        kwargs['batch_first'] = True

        if kwargs.get('is_label') is True:
            kwargs['sequential'] = False
            kwargs['unk_token'] = None
            kwargs['is_target'] = True

        else:
            if kwargs.get('preprocessing') is None:
                kwargs['preprocessing'] = lambda seq: [
                    self.SYM_SOS] + seq + [self.SYM_EOS]
            else:
                function = kwargs['preprocessing']
                kwargs['preprocessing'] = lambda seq: [
                    self.SYM_SOS] + function(seq) + [self.SYM_EOS]

            self.sos_id = None
            self.eos_id = None
        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super().build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]
