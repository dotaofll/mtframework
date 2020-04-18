import torch
from torch.autograd import Variable

class Predictor(object):

    def __init__(self,model,src_vocab,tgt_vocab):
        if torch.cuda.is_available():
            self.model = model.cuda()
        self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def get_decoder_features(self,src_seq):
        src_seq2tensor = torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]).view(1,-1)
        if torch.cuda.is_available():
            src_seq2tensor = src_seq2tensor.cuda()

        with torch.no_grad():
            softmx_list,_,other = self.model(src_seq2tensor,len(src_seq2tensor))
        return other

    def predict(self,src_seq):
        other = self.get_decoder_features(src_seq)

        length = other['length'][0]

        tgt_tensor = [other['sequence'][di][0] for di in range(length)]
        tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_tensor]
        return tgt_seq

        def predict_n(self, src_seq, n=1):
            
            other = self.get_decoder_features(src_seq)

            result = []
            for x in range(0, int(n)):
                length = other['topk_length'][0][x]
                tgt_id_seq = [other['topk_sequence'][di][0, x, 0].data[0] for di in range(length)]
                tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
                result.append(tgt_seq)

            return result  