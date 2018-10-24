import argparse
from datetime import datetime
from os.path import isabs, join
from docqa import model_dir
from docqa import trainer
from docqa.data_processing.document_splitter import MergeParagraphs, ShallowOpenWebRanker
from docqa.data_processing.multi_paragraph_qa import StratifyParagraphsBuilder, \
    StratifyParagraphSetsBuilder, RandomParagraphSetDatasetBuilder
from docqa.data_processing.preprocessed_corpus import PreprocessedData
from docqa.data_processing.qa_training_data import ContextLenBucketedKey
from docqa.dataset import ClusteredBatcher
from docqa.evaluator import LossEvaluator, MultiParagraphSpanEvaluator
from docqa.scripts.ablate_triviaqa import get_model
from docqa.text_preprocessor import WithIndicators
from docqa.trainer import SerializableOptimizer, TrainParams
from docqa.triviaqa.build_span_corpus import TriviaQaOpenDataset
from docqa.triviaqa.training_data import ExtractMultiParagraphsPerQuestion
from multiqa_infra.logger import ElasticLogger


def main():
    parser = argparse.ArgumentParser(description='Train a model on TriviaQA unfiltered')
    parser.add_argument('mode', choices=["confidence", "merge", "shared-norm",
                                         "sigmoid", "paragraph"])
    parser.add_argument("name", help="Where to store the model")
    parser.add_argument("-t", "--n_tokens", default=400, type=int,
                        help="Paragraph size")
    parser.add_argument('-n', '--n_processes', type=int, default=2,
                        help="Number of processes (i.e., select which paragraphs to train on) "
                             "the data with"
                        )
    parser.add_argument("-s", "--source_dir", type=str, default=None,
                        help="where to take input files")
    parser.add_argument("--n_epochs", type=int, default=None,
                        help="Max number of epoches to train on ")
    parser.add_argument("--char_th", type=int, default=None,
                        help="char level embeddings")
    parser.add_argument("--hl_dim", type=int, default=None,
                        help="hidden layer dim size")
    parser.add_argument("--regularization", type=int, default=None,
                        help="hidden layer dim size")
    parser.add_argument("--LR", type=float, default=1.0,
                        help="hidden layer dim size")
    parser.add_argument("--init_from", type=str, default=None,
                        help="model to init from")
    args = parser.parse_args()
    mode = args.mode

    #out = args.name + "-" + datetime.now().strftime("%m%d-%H%M%S")
    out = join('models',args.name)

    char_th = 100
    hl_dim = 140
    if args.char_th is not None:
        print(args.char_th)
        char_th = int(args.char_th)
        out += '--th' + str(char_th)
    if args.hl_dim is not None:
        print(args.hl_dim)
        hl_dim = int(args.hl_dim)
        out += '--hl' + str(hl_dim)

    if args.init_from is None:
        model = get_model(char_th, hl_dim, mode, WithIndicators())
    else:
        model_dir1 = model_dir.ModelDir(args.init_from)
        model = model_dir1.get_model()

    extract = ExtractMultiParagraphsPerQuestion(MergeParagraphs(args.n_tokens), ShallowOpenWebRanker(16),
                                                model.preprocessor, intern=True)

    eval = [LossEvaluator(), MultiParagraphSpanEvaluator(8, "triviaqa", mode != "merge", per_doc=False)]
    oversample = [1] * 4

    if mode == "paragraph":
        n_epochs = 120
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True),
                                          oversample,  only_answers=True)
    elif mode == "confidence" or mode == "sigmoid":
        if mode == "sigmoid":
            n_epochs = 640
        else:
            n_epochs = 160
        test = RandomParagraphSetDatasetBuilder(120, "flatten", True, oversample)
        train = StratifyParagraphsBuilder(ClusteredBatcher(60, ContextLenBucketedKey(3), True), oversample)
    else:
        n_epochs = 80
        test = RandomParagraphSetDatasetBuilder(120, "merge" if mode == "merge" else "group", True, oversample)
        train = StratifyParagraphSetsBuilder(30, mode == "merge", True, oversample)

    if args.n_epochs is not None:
        n_epochs = args.n_epochs
        out += '--' + str(n_epochs)


    data = TriviaQaOpenDataset(args.source_dir)

    async_encoding = 10
    #async_encoding = 0
    params = TrainParams(
        SerializableOptimizer("Adadelta", dict(learning_rate=args.LR)),
        num_epochs=n_epochs,num_of_steps=300000, ema=0.999, max_checkpoints_to_keep=2,
        async_encoding=async_encoding, log_period=30, eval_period=1800, save_period=1800,
        eval_samples=dict(dev=None, train=6000),regularization_weight=None
    )

    data = PreprocessedData(data, extract, train, test, eval_on_verified=False)

    data.preprocess(args.n_processes, 1000)

    with open(__file__, "r") as f:
        notes = f.read()
    notes = "Mode: " + args.mode + "\n" + notes

    if args.init_from is not None:
        init_from = model_dir.ModelDir(args.init_from).get_best_weights()
        if init_from is None:
            init_from = model_dir.ModelDir(args.init_from).get_latest_checkpoint()
    else:
        init_from = None

    trainer.start_training(data, model, params, eval, model_dir.ModelDir(out), notes, initialize_from=init_from)


if __name__ == "__main__":
    main()