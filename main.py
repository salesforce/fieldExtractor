from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import torch
from torch.nn import CrossEntropyLoss
from transformers import (
    BertTokenizer,
)

from models import LayoutlmConfig,FieldExtractor

from evaluate import evaluate

logger = logging.getLogger(__name__)

def get_labels(path):
    with open(path, "r") as f:
        labels = f.read().splitlines()
    return labels

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        default='datasets/',
        type=str,
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="the maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_lower_case",
        default=True,
        help="set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="avoid using CUDA when available"
    )
    parser.add_argument(
        "--test_mode", type=str, default="inv_cdip_test", help="name of test dataset"
    )
    parser.add_argument(
        "--online_refine_weight", type=float, default=1.0, help="online refine weight"
    )
    parser.add_argument(
        "--branch_num", type=int, default=3, help="number of branches for refinement"
    )
    parser.add_argument(
        "--log_name", type=str, default="eval.log"
    )
    parser.add_argument("--labels_name", default='labels.txt', type=str)

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, args.log_name),
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    labels = get_labels(os.path.join(args.data_dir, args.labels_name)) # fields of interest
    num_labels = len(labels)
    # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
    pad_token_label_id = CrossEntropyLoss().ignore_index

    config_class, model_class, tokenizer_class = LayoutlmConfig, FieldExtractor, BertTokenizer
    config = config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=None,
    )
    config.online_refine_weight = args.online_refine_weight
    config.branch_num = args.branch_num
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=None,
    )

    model.to(args.device)
    # Evaluation
    results = {}

    tokenizer = tokenizer_class.from_pretrained(
        args.model_name_or_path, do_lower_case=args.do_lower_case
    )
    checkpoints = [args.model_name_or_path]
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:

        model = model_class.from_pretrained(checkpoint)
        model.to(args.device)
        result = evaluate(
                args,
                model,
                tokenizer,
                labels,
                pad_token_label_id,
                mode=args.test_mode,
        )

        results.update(result)

    for key in sorted(results.keys()):
        print("{} = {}\n".format(key, str(results[key])))

    return results


if __name__ == "__main__":
    main()
