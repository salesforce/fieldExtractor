from __future__ import absolute_import, division, print_function

import os
import logging
from scipy.special import softmax
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm
import pickle as pkl

from datasets import INV_CDIP

logger = logging.getLogger(__name__)

def compute_eval_metrics(args, eval_loss, preds_list, labels, mode):

    def _load_gt_pairs(label_file_path):
        field_labels = dict()
        with open(label_file_path, encoding="utf-8") as fl:
            for lline in fl:
                lsplits = lline.split("\t")
                if lsplits[0] in ['', ' ']:
                    continue

                if len(lsplits) < 2:
                    continue
                fname = lsplits[-1].strip()
                field_label = lsplits[2].strip()
                if not fname in field_labels:
                    field_labels[fname] = dict()
                if not field_label in field_labels[fname]:
                    field_labels[fname][field_label] = []
                field_labels[fname][field_label].append(lsplits[0])

        return field_labels

    def _calc_tag_accuracy(pred_pairs, tag, gt_pairs, case_sensitive=False):
        tp, fp, fn = 0, 0, 0
        found = False

        for key_name in pred_pairs:
            if key_name == 'other':
                continue

            if key_name == tag:
                if tag not in gt_pairs:
                    fp += 1
                    continue

                tag_labels = gt_pairs[tag]
                tag_labels = [a.replace('"', '') for a in tag_labels]
                tag_labels = [a.replace(' ', '') for a in tag_labels]
                tag_labels = [a.replace('\n', '') for a in tag_labels]
                tag_labels = [a.replace('\r', '') for a in tag_labels]
                tag_labels = [a.replace(',', '') for a in tag_labels]

                text = pred_pairs[tag].replace('"', '').replace(' ', '')
                text = text.replace('\n', '').replace('\r', '').replace(',', '')

                if not case_sensitive:
                    tag_labels = [a.lower() for a in tag_labels]
                    text = text.lower()

                if text in tag_labels:
                    tp += 1
                    found = True
                else:

                    fp += 1
        if not found:
            if tag in gt_pairs:
                tag_labels = gt_pairs[tag]
                tag_labels = [a.replace('"', '') for a in tag_labels]
                tag_labels = [a.replace(' ', '') for a in tag_labels]
                tag_labels = [a.replace('\n', '') for a in tag_labels]
                tag_labels = [a.replace('\r', '') for a in tag_labels]
                tag_labels = [a.replace(',', '') for a in tag_labels]

                if (len(tag_labels) == 0) and (tag not in pred_pairs):
                    tp += 1
                else:
                    fn += 1

        return tp, fp, fn

    def _evaluate_field_end2end(tags, pred_pairs_all, gt_pairs_all):
        for tag in tags:
            vars()[tag.lower() + '_tp'] = 0
            vars()[tag.lower() + '_fp'] = 0
            vars()[tag.lower() + '_fn'] = 0

        for image_path in pred_pairs_all:
            if image_path not in gt_pairs_all:
                continue

            gt_pairs = gt_pairs_all[image_path]
            pred_pairs = pred_pairs_all[image_path]

            for tag in tags:
                tp, fp, fn = _calc_tag_accuracy(pred_pairs, tag, gt_pairs)
                vars()[tag.lower() + '_tp'] += tp
                vars()[tag.lower() + '_fp'] += fp
                vars()[tag.lower() + '_fn'] += fn

        results = {'precision': {}, 'recall': {}, 'f1 score': {}}

        precisions = []
        recalls = []
        f1s = []
        for tag in tags:
            precision = vars()[tag.lower() + '_tp'] / max(
                1e-10, float(vars()[tag.lower() + '_tp'] + vars()[tag.lower() + '_fp']))
            recall = vars()[tag.lower() + '_tp'] / max(
                1e-10, float(vars()[tag.lower() + '_tp'] + vars()[tag.lower() + '_fn']))
            f1 = 2. * precision * recall / max(1e-10, precision + recall)

            results['precision'][tag] = precision
            results['recall'][tag] = recall
            results['f1 score'][tag] = f1
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

        return results

    def _eval_field_level_pred(args, eval_loss, preds_list, labels, mode, threshold=0.1):
        # pick one value for each field
        preds = []
        probs = []
        fnames = []
        texts = []
        for pred in preds_list:
            texts.append(pred[0])
            preds.append(pred[1])
            probs.append(pred[2])
            fnames.append(pred[3])

        # load ground-truth
        label_file_path = os.path.join(args.data_dir, "{}_labels.txt".format(mode))
        gt_pairs_all = _load_gt_pairs(label_file_path)
        # predicted value for each field
        pred_pairs_all = dict()
        # predicted score of the value for each field
        pred_scores_all = dict()
        results_per_field = dict()

        field_list_gt = []
        for fn_ in gt_pairs_all:
            field_list_gt.extend(list(gt_pairs_all[fn_].keys()))
        field_list_gt = list(set(field_list_gt))
        field_list_pred = []
        for p in preds:
            field_list_pred.append(p)
        field_list_pred = list(set(field_list_pred))
        field_list_gt.extend(field_list_pred)
        field_list_to_eval = list(set(field_list_gt))
        # evaluated fields is a superset of fields in GT and fields in preds
        field_list_to_eval = [fd for fd in field_list_to_eval if fd in labels and not fd in ['O' , 'background']]
        for fi, fn in enumerate(fnames):
            if not fn in pred_pairs_all:
                pred_pairs_all[fn] = dict()
                pred_scores_all[fn] = dict()
            for fd in field_list_to_eval:
                fd_index = labels.index(fd)
                prob_score = probs[fi][fd_index]

                if fd in pred_pairs_all[fn] and prob_score <= pred_scores_all[fn][fd]:
                    continue
                pred_pairs_all[fn][fd] = texts[fi]
                pred_scores_all[fn][fd] = prob_score

        # remove predictions that lower than threshold
        fns = list(pred_pairs_all.keys())
        for fn in fns:
            fds = list(pred_pairs_all[fn].keys())
            for fd in fds:
                if pred_scores_all[fn][fd] < threshold:
                    del pred_pairs_all[fn][fd]
                    del pred_scores_all[fn][fd]
        end2end_results = _evaluate_field_end2end(field_list_to_eval, pred_pairs_all, gt_pairs_all)
        for fd_ in field_list_to_eval:
            if not fd_ in results_per_field:
                results_per_field[fd_] = dict()
            results_per_field[fd_]['f1 score'] = end2end_results['f1 score'][fd_]
            results_per_field[fd_]['precision'] = end2end_results['precision'][fd_]
            results_per_field[fd_]['recall'] = end2end_results['recall'][fd_]

        end2end_macro_average_f1 = float(sum(end2end_results['f1 score'].values())) / len(end2end_results['f1 score'])
        end2end_macro_average_precision = float(sum(end2end_results['precision'].values())) / len(end2end_results['precision'])
        end2end_macro_average_recall = float(sum(end2end_results['recall'].values())) / len(end2end_results['recall'])

        results = {
            "loss": eval_loss,
            "end2end marco average precision": end2end_macro_average_precision,
            "end2end marco average recall": end2end_macro_average_recall,
            "end2end marco average f1": end2end_macro_average_f1,
        }

        return results, pred_pairs_all, results_per_field

    results, pred_pairs_all, results_per_field = _eval_field_level_pred(args, eval_loss, preds_list, labels, mode)

    return results, pred_pairs_all, results_per_field

def _post_processing_no_grouping(words, pred_scores, field_list_set):
    phrase_list = []
    pred_fields = []
    pred_scores_list = []
    pred_scores = np.array(pred_scores)
    preds = np.argmax(pred_scores, axis=-1)
    i = 0
    while (i < preds.shape[0]):
        p = preds[i]
        phrase_list.append(words[i])
        pred_fields.append(field_list_set[p])
        pred_scores_list.append(pred_scores[i])
        i += 1

    return phrase_list, pred_fields, pred_scores_list

def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode):
    eval_dataset = INV_CDIP(args, tokenizer, labels, pad_token_label_id, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=None,
    )
    field_list_set = labels

    logger.info("***** Running evaluation %s *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0

    model.eval()
    preds_lists = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].to(args.device),
                "attention_mask": batch[1].to(args.device),
                "token_type_ids": batch[2].to(args.device),
                "labels": batch[3].to(args.device),
                "bbox": batch[4].to(args.device),
            }

            outputs = model(**inputs)

            tmp_eval_loss, logits = outputs[0], outputs[1]

            for b_num in range(2, args.branch_num+1):
                logits += outputs[b_num]

            logits /= (args.branch_num*1.0)

            filenames = batch[5]
            tokens = batch[6]
            if args.n_gpu > 1:
                tmp_eval_loss = (
                    tmp_eval_loss.mean()
                )  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()

        nb_eval_steps += 1

        batch_size = logits.shape[0]
        token_num = logits.shape[1]
        scores_all = logits.detach().cpu().numpy()
        labels_idx = inputs["labels"].detach().cpu().numpy()
        for bs in range(0, batch_size):
            fn = filenames[bs]
            words = []
            pred_scores = []
            word_labels = []
            for tk in range(0, token_num):
                lb_idx = labels_idx[bs][tk]
                if lb_idx == pad_token_label_id:
                    continue
                pred_scores.append(softmax(scores_all[bs][tk]))
                words.append(tokens[tk][bs])
                word_labels.append(lb_idx)
            token_list, pred_fields, pred_scores_list = _post_processing_no_grouping(words, pred_scores, field_list_set)
            for t, p, ps in zip(token_list, pred_fields, pred_scores_list):
                preds_lists.append((t, p, ps, fn))

    eval_loss = eval_loss / nb_eval_steps

    end2end_results, pred_pairs, results_per_field = compute_eval_metrics(args, eval_loss, preds_lists, field_list_set, mode)

    with open(os.path.join(args.output_dir, 'prediction_pairs.pkl'), 'wb') as f:
        pkl.dump(pred_pairs, f)

    results = {
        "loss": eval_loss,
        "end2end marco average precision": end2end_results["end2end marco average precision"],
        "end2end marco average recall": end2end_results["end2end marco average recall"],
        "end2end marco average f1": end2end_results["end2end marco average f1"],
    }

    logger.info("***** Eval results %s *****")
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results

