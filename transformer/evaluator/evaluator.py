import torch
import torchtext
from transformer.loss.loss import NLLLoss


class Evaluator(object):
    """
    Class to evaluate models with given datasets.

    Args:
        loss (transformer.loss.loss.NLLLoss): loss for evaluator (default: transformer.loss.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)
    """

    def __init__(self, loss=NLLLoss(), batch_size=64):
        self.loss = loss
        self.batch_size = batch_size
        self.max_length = 120

    def evaluate(self, model, data):
        """
        Evaluate a model on given dataset and return performance.

        Args:
            model (transformer.transformer): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against

        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields['tgt'].vocab
        pad = tgt_vocab.stoi[data.fields['tgt'].pad_token]

        with torch.no_grad():
            for batch in batch_iterator:
                encoder_inputs, _ = getattr(batch, 'src')
                targets = getattr(batch, 'tgt')

                enc_outputs, enc_self_attns = self.model.encoder(encoder_inputs)
                decoder_inputs = torch.zeros(1, 5).type_as(encoder_inputs.data)
                next_symbol = tgt_vocab.stoi['<sos>']

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

        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return accuracy
