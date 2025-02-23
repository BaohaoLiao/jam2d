import random
import os
import argparse
import time
from datetime import datetime
from tqdm import tqdm
from collections import Counter
from vllm import LLM, SamplingParams

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify, ExprExtractionConfig

from external.qwen25_math_evaluation.evaluate import evaluate
from external.qwen25_math_evaluation.utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from external.qwen25_math_evaluation.parser import *
from external.qwen25_math_evaluation.trajectory import *
from external.qwen25_math_evaluation.data_loader import load_data
from external.qwen25_math_evaluation.python_executor import PythonExecutor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_names", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--model_name_or_path", default="gpt-4", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--prompt_type", default="tool-integrated", type=str)
    parser.add_argument("--split", default="test", type=str)
    parser.add_argument("--num_test_sample", default=-1, type=int)  # -1 for full data
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--n_sampling", default=1, type=int)
    parser.add_argument("--top_p", default=1, type=float)
    parser.add_argument("--max_tokens_per_call", default=2048, type=int)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--save_outputs", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--use_safetensors", action="store_true")
    parser.add_argument("--num_shots", type=int, default=0)
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Apply chat template to prompt.",
    )
    parser.add_argument("--pipeline_parallel_size", type=int, default=1)
    parser.add_argument(
        "--adapt_few_shot",
        action="store_true",
        help="Few shot for multiple-choice questions, zero shot for others.",
    )

    # 2D maj
    parser.add_argument("--num_think_chunks", default=1, type=int)
    parser.add_argument("--max_tokens_per_answer", default=512, type=int)

    args = parser.parse_args()
    args.top_p = (
        1 if args.temperature == 0 else args.top_p
    )  # top_p must be 1 when using greedy sampling (vllm)
    return args


def prepare_data(data_name, args):
    if "math500_level" in data_name:
        level = int(data_name.strip()[-1])
        examples = load_data("math500", args.split, args.data_dir)
        examples = [example for example in examples if example["level"]==level]
    else:
        examples = load_data(data_name, args.split, args.data_dir)

    # sample `num_test_sample` from dataset
    if args.num_test_sample > 0:
        # examples = random.sample(examples, min(args.num_test_sample, len(examples)))
        examples = examples[: args.num_test_sample]

    # shuffle
    if args.shuffle:
        random.seed(datetime.now().timestamp())
        random.shuffle(examples)

    # select start and end
    examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    out_file_prefix = f"{args.split}_{args.prompt_type}_{args.num_test_sample}_seed{args.seed}_t{args.temperature}"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    out_file = f"{output_dir}/{data_name}/{out_file_prefix}_s{args.start}_e{args.end}_sampling{args.n_sampling}.jsonl"
    os.makedirs(f"{output_dir}/{data_name}", exist_ok=True)

    return examples, out_file


def setup(args):
    # load model
    available_gpus = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=len(available_gpus) // args.pipeline_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        trust_remote_code=True,
    )
    tokenizer = llm.get_tokenizer()

    # infer & eval
    data_list = args.data_names.split(",")
    results = []
    for data_name in data_list:
        results.append(main(llm, tokenizer, data_name, args))

    # add "avg" result to data_list and results
    if args.n_sampling == 1:
        print("accuracy:")
        data_list.append("avg")
        results.append(
            {
                "acc": sum([result["acc"] for result in results]) / len(results),
            }
        )

        # print all results
        pad = max([len(data_name) for data_name in data_list])
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['acc']:.2f}".ljust(pad, " ") for result in results]))
    else:
        print(f"maj@{args.n_sampling} accuracy:")
        data_list.append("avg")
        results.append(
            {
                "maj_acc": sum([result["maj_acc"] for result in results]) / len(results),
            }
        )

        # print all results
        pad = max([len(data_name) for data_name in data_list])
        print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
        print("\t".join([f"{result['maj_acc']:.2f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def parse_gt(example, data_name):
    parsed_gt_ans = parse(
        example["gt_cot"],
        extraction_config=[
            LatexExtractionConfig(
                boxed_match_priority=0,
                try_extract_without_anchor=True,
            ),
        ]
    )
    assert len(parsed_gt_ans) > 0
    return parsed_gt_ans


def extract_pred_and_parse(code, data_name):
    if "boxed" in code:
        pred = parse(
            code, 
            extraction_config=[
                LatexExtractionConfig(
                    boxed_match_priority=0, 
                    try_extract_without_anchor=True,
                ),
            ]
        )
        return pred
    else:
        return []


def get_most_common_pred_score(preds, scores):
    valid_pairs = [(pred, score) for pred, score in zip(preds, scores) if pred != ""]
    if not valid_pairs:
        return "", False
    
    valid_preds = [pair[0] for pair in valid_pairs]
    most_common_pred = Counter(valid_preds).most_common(1)[0][0]
    for pred, score in valid_pairs:
        if pred == most_common_pred:
            return pred, score
    return "", False


def obtain_scores(samples, data_name, n_sampling=1):
    all_samples = []
    correctnesses = []
    for sample in samples:
        scores = [verify(sample["pred"][i], sample["gt"]) for i in range(len(sample["pred"]))]
        correctnesses.append(scores[0])
        sample.update({"score": scores})

        orig_preds = sample["pred"]
        orig_gt = sample["gt"]
        sample.pop("pred")
        sample.pop("gt")

        new_preds = []
        for i in range(len(orig_preds)):
            if scores[i]:
                new_preds.append(str(orig_gt[0]))
            else:
                if orig_preds[i]:
                    new_preds.append(str(orig_preds[i][0]))
                else:
                    new_preds.append("")

        sample.update({
            "gt": str(orig_gt[0]),
            "pred": new_preds
        })
        all_samples.append(sample)

    result_json = {
        "num_samples": len(correctnesses),
        "acc": float(f"{sum(correctnesses) / len(correctnesses):.4f}") * 100,
    }

    if n_sampling > 1:
        new_all_samples = []
        maj_correctnesses = []
        for sample in all_samples:
            maj_pred, maj_score = get_most_common_pred_score(sample["pred"], sample["score"])
            sample.update({"maj_pred": maj_pred, "maj_score": maj_score})
            new_all_samples.append(sample)
            maj_correctnesses.append(maj_score)

        result_json["maj_acc"] = float(f"{sum(maj_correctnesses) / len(maj_correctnesses):.4f}") * 100
        all_samples = new_all_samples

    return all_samples, result_json


def obtain_2d_sub_scores_and_preds(gt, sub_preds):
    sub_scores = []
    for sub_pred in sub_preds:
        sub_scores.append([verify(gt, pred) for pred in sub_pred])

    new_gt =str(gt[0])
    new_sub_preds = []
    for i, sub_pred in enumerate(sub_preds):
        new_sub_pred = []
        for j, score in enumerate(sub_scores[i]):
            if score:
                new_sub_pred.append(new_gt)
            else:
                if sub_pred[j]:
                    new_sub_pred.append(str(sub_pred[j][0]))
                else:
                    new_sub_pred.append("")
        new_sub_preds.append(new_sub_pred)

    maj_preds = []
    maj_scores = []
    for preds, scores in zip(new_sub_preds, sub_scores):
        pred, score = get_most_common_pred_score(preds, scores)
        maj_preds.append(pred)
        maj_scores.append(score)
    
    return new_gt, new_sub_preds, sub_scores, maj_preds, maj_scores


def obtain_2d_scores(samples, data_name, n_sampling=1):
    all_samples = []
    correctnesses = []
    for sample in samples:
        scores = sample["score"]
        correctnesses.append(scores[0])
        all_samples.append(sample)

    result_json = {
        "num_samples": len(correctnesses),
        "acc": float(f"{sum(correctnesses) / len(correctnesses):.4f}") * 100,
    }

    if n_sampling > 1:
        new_all_samples = []
        maj_correctnesses = []
        for sample in all_samples:
            print("###", sample["sub_preds"], sample["sub_scores"], sample["pred"], sample["score"])
            maj_pred, maj_score = get_most_common_pred_score(sample["pred"], sample["score"])
            sample.update({"maj_pred": maj_pred, "maj_score": maj_score})
            new_all_samples.append(sample)
            maj_correctnesses.append(maj_score)

        result_json["maj_acc"] = float(f"{sum(maj_correctnesses) / len(maj_correctnesses):.4f}") * 100
        all_samples = new_all_samples
    return all_samples, result_json


def main(llm, tokenizer, data_name, args):
    examples, out_file = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " , #samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    samples = []
    for i, example in tqdm(enumerate(examples), total=len(examples)):
        idx = example["idx"]

        # parse question and answer
        example["question"] = parse_question(example, data_name)
        if example["question"] == "":
            continue
        gt_cot, _ = parse_ground_truth(example, data_name)
        example["gt_cot"] = gt_cot
        gt_ans = parse_gt(example, data_name)
        
        full_prompt = construct_prompt(example, data_name, args)

        if i == args.start:
            print(full_prompt)

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
            "prompt": full_prompt,
        }

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    # repeat n times
    input_prompts = [sample["prompt"] for sample in samples for _ in range(args.n_sampling)]
    if args.apply_chat_template:
        input_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt.strip()}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in input_prompts
        ]
    current_prompts = [(i, prompt) for i, prompt in enumerate(input_prompts)]

    # start inference
    start_time = time.time()
    # get all outputs
    prompts = [item[1] for item in current_prompts]
    outputs = llm.generate(
        prompts,
        SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens_per_call,
            n=1,
        ),
    )
    outputs = sorted(outputs, key=lambda x: int(x.request_id))  # sort outputs by request_id
    codes = [output.outputs[0].text for output in outputs]
    assert len(codes) == len(current_prompts)

    # original
    if args.num_think_chunks == 1:
        # extract preds
        results = [extract_pred_and_parse(code, data_name) for code in codes]
        time_use = time.time() - start_time

        # put results back to examples
        all_samples = []
        for i, sample in enumerate(samples):
            code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
            preds = results[i * args.n_sampling : (i + 1) * args.n_sampling]
            sample.pop("prompt")
            sample.update({"completion": code, "pred": preds})
            all_samples.append(sample)

        # add processed samples
        all_samples, result_json = obtain_scores(
            samples=all_samples, 
            data_name=data_name,
            n_sampling=args.n_sampling,
        )

    # 2D majority voting
    else:
        ori_think_sums = []
        for code in codes:
            if len(code.split("</think>")) == 1:
                ori_think_sums.append(code)
            else:
                ori_think_sums.append(code.split("</think>")[-1])
        
        reasonings_tok = [tokenizer.encode(code.split("</think>")[0])[1:] for code in codes]
        new_prompts = []
        for r, reasoning in enumerate(reasonings_tok):
            splits = [reasoning[: i * len(reasoning) // args.num_think_chunks] for i in range(1, args.num_think_chunks)]  # cut evenly
            new_prompts.extend([prompts[r] + tokenizer.decode(split) + "\n</think>\n\n" for split in splits])

        new_outputs = llm.generate(
            new_prompts,
            SamplingParams(
                temperature=0,
                top_p=1,
                max_tokens=args.max_tokens_per_answer,
                n=1,
            ),
        )
        new_outputs = sorted(new_outputs, key=lambda x: int(x.request_id))
        inter_think_sums = [output.outputs[0].text for output in new_outputs]
        assert len(inter_think_sums) == len(new_prompts)

        all_think_sums = []
        for i in range(len(ori_think_sums)):
            all_think_sums.extend(
                 inter_think_sums[i*(args.num_think_chunks-1) : (i+1)*(args.num_think_chunks-1)] + [ori_think_sums[i]]
            )
        all_results = [extract_pred_and_parse(think_sum, data_name) for think_sum in all_think_sums]
        time_use = time.time() - start_time

        # put results back to examples
        all_samples = []
        for i, sample in enumerate(samples):
            code = codes[i * args.n_sampling : (i + 1) * args.n_sampling]
            think_sums = all_think_sums[i*args.n_sampling*args.num_think_chunks : (i+1)*args.n_sampling*args.num_think_chunks]
            sample_think_sums = [think_sums[n*args.num_think_chunks:(n+1)*args.num_think_chunks] for n in range(args.n_sampling)]
            preds = all_results[i*args.n_sampling*args.num_think_chunks : (i+1)*args.n_sampling*args.num_think_chunks]
            sample_preds = [preds[n*args.num_think_chunks:(n+1)*args.num_think_chunks] for n in range(args.n_sampling)]
            sample.pop("prompt")


            new_gt, sub_preds, sub_scores, maj_preds, maj_scores = obtain_2d_sub_scores_and_preds(sample["gt"], sample_preds)
            sample.pop("gt")
            sample.update({
                "completion": code,
                "think_sums":sample_think_sums,
                "gt": new_gt,
                "sub_preds": sub_preds,
                "sub_scores": sub_scores,
                "pred": maj_preds,
                "score": maj_scores,
            })
            all_samples.append(sample)

        # add processed samples
        all_samples, result_json = obtain_2d_scores(
            samples=all_samples, 
            data_name=data_name,
            n_sampling=args.n_sampling,
        )


    # save outputs
    save_jsonl(all_samples, out_file)

    result_json["time_use_in_second"] = time_use
    result_json["time_use_in_minite"] = (
        f"{int(time_use // 60)}:{int(time_use % 60):02d}"
    )

    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    setup(args)
