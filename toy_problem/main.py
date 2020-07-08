import os
import argparse
import logging
import torch
import torchtext
import sys
import warnings
sys.path.append('..')
from transformer.checkpoint.checkpoint import Checkpoint
from transformer.dataset.field import SourceField, TargetField
from transformer.evaluator.predictor import Predictor
from transformer.loss.loss import Perplexity, NLLLoss
from transformer.models.transformer import Transformer
from transformer.trainer.supervised_trainer import SupervisedTrainer


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', action='store', dest='train_path',
                        help='Path to train data')
    parser.add_argument('--dev_path', action='store', dest='dev_path',
                        help='Path to dev data')
    parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                        help='Path to experiment directory. '
                             'If load_checkpoint is True, then path to checkpoint directory has to be provided')
    parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                        help='The name of the checkpoint to load, usually an encoded time string')
    parser.add_argument('--resume', action='store_true', dest='resume',
                        default=False,
                        help='Indicates if training has to be resumed from the latest checkpoint')
    parser.add_argument('--log-level', dest='log_level',
                        default='info',
                        help='Logging level.')
    parser.add_argument('--d_model', type=int,
                        default=512,
                        help='dimension of transformer model')
    parser.add_argument('--num_heads', type=int,
                        default=8,
                        help='number of attention heads')
    parser.add_argument('--d_ff', type=int,
                        default=2048,
                        help='dimension of position-wise feed forward network')
    parser.add_argument('--num_encoder_layers', type=int,
                        default=6,
                        help='number of encoder layers')
    parser.add_argument('--num_decoder_layers', type=int,
                        default=6,
                        help='number of decoder layers')
    parser.add_argument('--dropout_p', type=float,
                        default=0.3,
                        help='dropout probability')
    parser.add_argument('--ffnet_style', type=str,
                        default='ff',
                        help='position-wise feed forward network style [ff, conv]')

    opt = parser.parse_args()

    LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
    logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
    logging.info(opt)

    if opt.load_checkpoint is not None:
        logging.info("loading checkpoint from {}".format(
            os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint))
        )
        checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
        checkpoint = Checkpoint.load(checkpoint_path)
        model = checkpoint.model
        input_vocab = checkpoint.input_vocab
        output_vocab = checkpoint.output_vocab
    else:
        # Prepare dataset
        src = SourceField()
        tgt = TargetField()
        max_len = 50


        def len_filter(example):
            return len(example.src) <= max_len and len(example.tgt) <= max_len


        train = torchtext.data.TabularDataset(
            path=opt.train_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=len_filter
        )
        dev = torchtext.data.TabularDataset(
            path=opt.dev_path, format='tsv',
            fields=[('src', src), ('tgt', tgt)],
            filter_pred=len_filter
        )
        src.build_vocab(train, max_size=50000)
        tgt.build_vocab(train, max_size=50000)
        input_vocab = src.vocab
        output_vocab = tgt.vocab

        # NOTE: If the source field name and the target field name
        # are different from 'src' and 'tgt' respectively, they have
        # to be set explicitly before any training or inference

        # Prepare loss
        weight = torch.ones(len(tgt.vocab))
        pad_id = tgt.vocab.stoi[tgt.pad_token]
        loss = NLLLoss(weight, pad_id)
        if torch.cuda.is_available():
            loss.cuda()

        model = None
        optimizer = None
        if not opt.resume:
            # Initialize model
            model = Transformer(len(tgt.vocab), pad_id, len(src.vocab), len(tgt.vocab), opt.d_model,
                                opt.d_ff, opt.num_heads, opt.num_encoder_layers, opt.num_decoder_layers,
                                opt.dropout_p, opt.ffnet_style)
            if torch.cuda.is_available():
                model.cuda()

            for param in model.parameters():
                param.data.uniform_(-0.08, 0.08)

            # Optimizer and learning rate scheduler can be customized by
            # explicitly constructing the objects and pass to the trainer.
            #
            # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
            # scheduler = StepLR(optimizer.optimizer, 1)
            # optimizer.set_scheduler(scheduler)

        # train
        t = SupervisedTrainer(loss=loss, batch_size=32,
                              checkpoint_every=5000,
                              print_every=10, expt_dir=opt.expt_dir)

        model = t.train(model, train, num_epochs=1, dev_data=dev, optimizer=optimizer, resume=opt.resume)

    predictor = Predictor(model, input_vocab, output_vocab)

    while True:
        seq_str = input("Type in a source sequence:")
        seq = seq_str.strip().split()
        print(predictor.predict(seq, tgt.sos_id))
