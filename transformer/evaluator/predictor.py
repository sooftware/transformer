import torch


class Predictor(object):

    def __init__(self, model, src_vocab, tgt_vocab):
        """
        Predictor class to evaluate for a given model.

        Args:
            model (transformer.models): trained model. This can be loaded from a checkpoint
                using `transformer.checkpoint.checkpoint.load`
            src_vocab (transformer.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (transformer.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        self.max_length = 120
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def predict(self, model, encoder_inputs, sos_id):
        """
        For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
        target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
        Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
        :param model: Transformer Model
        :param encoder_inputs: The encoder input
        :param sos_id: The start symbol. In this example it is 'S' which corresponds to index 4
        :return: The target input
        """
        enc_outputs, enc_self_attns = self.model.encoder(encoder_inputs)
        decoder_inputs = torch.zeros(1, 5).type_as(encoder_inputs.data)
        next_symbol = sos_id

        for i in range(self.max_length):
            decoder_inputs[0][i] = next_symbol
            decoder_outputs = self.model.decoder(decoder_inputs, encoder_inputs, enc_outputs)[0]
            projected = model.linear(decoder_outputs)
            prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            next_word = prob.data[i]
            next_symbol = next_word.item()

            print(next_symbol)
            if next_symbol == '<eos>':
                break

        return decoder_inputs

