"""

Convert the output of the MDLM model to the format expected by the SSD-LM evaluation script.

"""

# Current:
# {"text": "<|endoftext|>\n\nOnce upon a time, you only saw her\u2014your type of girl and guy. But life comes wit..."}

# Goal (for SSD-LM eval script):
# {"context_len": 5, "context": [50118, 50118, 133, 94, 86], "context_string": "\n\nThe last time", "len": 50, "tokens": [[1771, 261, 702, 11, 136, 5, 3445, 11, 315, 17, 27, 29, 3261, 21, 11, 6193, 6, 77, 37, 554, 411, 2856, 13, 5, 589, 9, 21544, 137, 3606, 10, 155, 12, 288, 872, 23, 2912, 196, 26399, 4, 50118, 50118, 20096, 261, 6, 54, 702, 13, 11946, 11, 5], [6, 52, 70, 2145, 5, 4463, 14, 439, 1593, 19, 3161, 315, 11, 5, 386, 9, 5, 336, 191, 4, 50118, 50118, 38778, 6, 38, 95, 218, 17, 27, 90, 236, 5, 1651, 11, 42, 1068, 6, 8, 38, 619, 6587, 59, 2198, 16099, 159, 10, 950, 142, 24, 34], [896, 8, 292, 97, 749, 2781, 4217, 9, 2398, 7, 1854, 21, 11, 4999, 4, 635, 6, 42, 76, 6, 8522, 33, 3604, 7, 671, 143, 8, 70, 2398, 51, 694, 7, 5, 997, 12, 25566, 247, 4, 50118, 50118, 5771, 5, 754, 14, 896, 5712, 251, 765, 9, 63], [14, 1102, 21, 11, 11735, 4, 20, 8903, 685, 316, 426, 14, 191, 19, 130, 9, 167, 426, 15, 5, 921, 14, 363, 137, 5, 191, 1249, 4, 178, 124, 172, 6, 5, 7839, 56, 49, 78, 339, 11, 2266, 137, 5, 191, 1249, 15, 5, 921, 4, 50118, 50118], [5, 2836, 177, 21, 702, 11, 3719, 21, 11, 10206, 4, 50118, 50118, 133, 5098, 32, 41, 20137, 2378, 54, 27701, 1782, 143, 1732, 9, 49, 586, 14, 473, 45, 3594, 5, 1374, 1298, 1651, 4, 252, 17, 27, 548, 57, 10, 538, 936, 11, 5, 375, 2202, 6, 45], [38, 794, 123, 6, 38, 21, 5283, 8, 37, 21, 1765, 13, 10, 4747, 908, 15, 1870, 4, 152, 16, 7371, 814, 31, 5, 313, 54, 26, 84, 3111, 58, 608, 205, 383, 8, 14, 37, 21, 66, 9, 39, 633, 142, 5, 29798, 362, 797, 9, 39, 3380, 4], [5, 3263, 12, 1000, 21, 278, 13, 10, 158, 12, 180, 1355, 21, 11, 5114, 6, 53, 14, 64, 4100, 95, 13207, 47, 11, 110, 9753, 6, 157, 4, 286, 7261, 6, 5, 12561, 7, 465, 10, 92, 512, 190, 114, 10, 158, 12, 180, 1355, 16, 122, 278, 15], [47, 1317, 9, 10, 5226, 14, 1074, 11, 5, 9099, 9, 2298, 852, 6, 235, 116, 20, 3157, 16, 6, 38, 218, 17, 27, 90, 4, 1234, 9, 70, 6, 24, 17, 27, 29, 95, 10, 1462, 809, 4, 318, 47, 95, 23829, 62, 5, 418, 14, 606, 31, 5], [51, 58, 259, 8, 38, 206, 24, 21, 5, 550, 4, 38, 95, 64, 17, 27, 90, 120, 5, 3493, 6, 70, 9, 5, 2356, 6, 9, 42, 1086, 317, 4, 38, 206, 51, 439, 11, 491, 6, 38, 1317, 79, 21, 66, 13, 5, 363, 19, 5, 1159, 6], [37, 21, 11, 127, 790, 6, 38, 21, 25, 4904, 25, 38, 21, 1220, 7, 28, 4, 91, 21, 11, 10, 14298, 6, 2498, 10, 1855, 337, 1073, 81, 127, 6085, 4, 38, 3179, 7, 173, 13, 123, 6, 142, 38, 21, 20085, 8, 5800, 4, 38, 21, 6023, 9], [51, 58, 45, 1165, 11, 10, 3107, 545, 177, 21, 11, 2338, 4, 50118, 50118, 2409, 6, 122, 6, 24, 18, 5, 4574, 54, 342, 5, 1164, 15, 5, 2805, 23, 316, 35, 612, 10, 4, 119, 4, 7008, 136, 3378, 6, 5, 1482, 9, 5, 502, 132, 2451, 14], [89, 21, 10, 6484, 872, 11, 10, 831, 2748, 6, 21, 11, 11265, 6, 77, 10, 2898, 831, 3627, 1447, 7, 1548, 5, 6444, 8, 21, 738, 159, 30, 470, 3054, 31274, 134, 742, 7806, 6, 5, 346, 9, 7446, 217, 4957, 30, 121, 4, 8449, 624, 155, 360, 6], [38, 794, 106, 6, 5, 94, 86, 79, 13356, 62, 6, 21, 818, 80, 688, 8, 10, 183, 137, 14, 6, 77, 38, 1317, 402, 11, 5, 17182, 250, 4, 20, 2369, 21, 77, 38, 6536, 106, 66, 6, 8, 38, 1798, 24, 456, 452, 4, 520, 79, 21, 66], [5, 232, 21, 15, 5, 97, 526, 21, 95, 601, 360, 536, 81, 132, 6, 151, 82, 4, 178, 25, 24, 1411, 31, 145, 10, 891, 9, 377, 7, 10, 881, 934, 559, 515, 9, 5, 76, 6, 24, 18, 1256, 543, 7, 4744, 5, 1683, 42, 40, 33, 15], [38, 399, 17, 27, 90, 697, 6, 38, 21, 45, 533, 7, 1597, 31, 5, 1160, 8, 21, 2114, 801, 4153, 744, 142, 9, 5, 2166, 8, 613, 810, 11, 6244, 10, 920, 4, 870, 122, 6, 70, 1791, 4395, 17, 27, 90, 33, 7, 28, 3711, 19, 10, 744], [79, 3244, 21, 11, 779, 4013, 4, 38, 21, 164, 11, 13, 10, 7402, 11, 5, 1692, 9, 4013, 4, 38, 4443, 23, 14, 477, 38, 18774, 350, 203, 59, 69, 4, 38, 1467, 38, 74, 33, 10, 1032, 19, 4690, 8, 14, 38, 115, 393, 652, 69, 4, 38], [38, 21, 15, 5, 882, 6, 37, 21, 546, 101, 37, 21, 820, 107, 793, 4, 91, 4021, 39, 1420, 81, 162, 8, 18626, 39, 124, 7, 39, 526, 19, 10, 4298, 6, 1618, 6941, 15, 127, 865, 77, 38, 478, 123, 4, 38, 56, 57, 11, 1511, 13, 291], [38, 56, 143, 2536, 1272, 880, 11, 3788, 6, 77, 127, 1141, 21, 3276, 7, 3951, 162, 19, 20449, 8456, 4, 96, 3788, 6, 127, 985, 21, 519, 10, 1144, 908, 8, 21, 848, 4, 9012, 14, 183, 6, 127, 1086, 301, 21, 5, 2373, 675, 9, 127, 301, 6], [37, 21, 3828, 15, 10, 13494, 3682, 6, 37, 21, 1428, 39, 308, 512, 8, 37, 314, 5, 882, 11, 760, 9, 5, 512, 8, 42, 621, 4024, 81, 8, 848, 4, 50118, 50118, 243, 18, 164, 7, 28, 182, 430, 62, 89, 6, 8, 38, 437, 164, 7, 28], [730, 34797, 62, 1703, 1272, 13, 961, 21, 361, 107, 536, 4, 635, 6, 452, 17, 27, 29, 1219, 13, 10, 1703, 27297, 21, 41, 908, 15, 127, 2721, 247, 4, 318, 47, 218, 17, 27, 90, 192, 42, 25, 10, 21279, 53, 10, 1846, 6, 172, 213, 124, 7]], "string": [" Waron played in against the Irish in United\u2019s Championship was in 1999, when he started six matches for the University of Ulster before suffering a 3-0 loss at Caledonian.\n\nWaron, who played for Brighton in the", ", we all remember the disaster that went wrong with Minnesota United in the start of the 2016 season.\n\nSadly, I just don\u2019t want the organization in this situation, and I feel terrible about completely shutting down a club because it has", " Canada and five other countries delivered supplies of weapons to Syria was in 2003. However, this year, Canadians have promised to return any and all weapons they provide to the war-torn country.\n\nWhile the fact that Canada falls long short of its", " that happened was in 1987. The Chargers lost 12 games that season with three of those games on the road that night before the season ended. And back then, the Vikings had their first win in 2008 before the season ended on the road.\n\n", " the championship game was played in Tennessee was in 1989.\n\nThe Tigers are an avid fan who hates seeing any version of their program that does not represent the overall winning organization. They\u2019ve been a major problem in the past decade, not", " I saw him, I was pregnant and he was calling for a chemical attack on Israel. This is radical action from the man who said our farmers were doing good things and that he was out of his job because the Arabs took control of his farm.", " the AT-X was set for a 10-year contract was in 1980, but that can apparently just scare you in your grave, well. For Toyota, the inability to find a new car even if a 10-year contract is now set on", " you heard of a journalist that lives in the shadow of Wall Street, right? The truth is, I don\u2019t. First of all, it\u2019s just a dead body. If you just suck up the money that comes from the", " they were here and I think it was the July. I just can\u2019t get the pictures, all of the photos, of this whole place. I think they went in News, I heard she was out for the night with the kids,", " he was in my house, I was as upset as I was allowed to be. He was in a wheelchair, wearing a chalag over my mouth. I refused to work for him, because I was ashamed and angry. I was afraid of", " they were not included in a Top 16 game was in 2009.\n\nAnd, now, it's the Lions who put the pressure on the USA at 12:00 a.m. PT against Philadelphia, the host of the June 2 draw that", " there was a fatal loss in a military connection, was in 1986, when a Japanese military ship failed to approach the ocean and was shot down by American aircraft.[1] Overall, the number of ships including destroyed by U. planes within 3 days,", " I saw them, the last time she woke up, was almost two weeks and a day before that, when I heard something in the POA. The sound was when I knocked them out, and I hear it again today. When she was out", " the world was on the other side was just 17 days ago over 2,000 people. And as it goes from being a couple of months to a single biggest political event of the year, it's pretty hard to imagine the effect this will have on", " I didn\u2019t live, I was not likely to die from the incident and was facing potential violent death because of the physical and financial risk in protecting a child. By now, all Americans shouldn\u2019t have to be threatened with a death", " she talked was in October 2005. I was going in for a divorce in the middle of 2005. I guess at that point I forgot too much about her. I knew I would have a fight with Elizabeth and that I could never face her. I", " I was on the field, he was looking like he was 22 years old. He threw his hands over me and flipped his back to his side with a motion, leaving tears on my hand when I hit him. I had been in contact for 20", " I had any mental problems began in 2000, when my wife was unable to treat me with psychiatric medication. In 2000, my mother was having a heart attack and was killed. Until that day, my whole life was the worst period of my life,", " he was convicted on a speeding ticket, he was driving his own car and he left the field in front of the car and this person drove over and killed.\n\nIt's going to be very different up there, and I'm going to be", " America screwed up traffic problems for everyone was 9 years ago. However, today\u2019s reason for a traffic debacle was an attack on my beautiful country. If you don\u2019t see this as a coincidence but a crime, then go back to"], "gold_tokens": [], "gold_string": ""}

import os
import glob
import json

import click

from transformers import AutoTokenizer


def get_possible_prompts(prompt_path):
    with open(prompt_path) as f:
        return [json.loads(line)["context_string"] for line in f]


def file_to_exp_info(file):
    parent_dir = os.path.dirname(file)
    info_path = os.path.join(parent_dir, 'info.json')
    with open(info_path) as f:
        relevant_config = json.load(f)['fk_steering']

    relevant_keys = [
        'potential_type',
        'k_particles',
        'lmbda',
        'reward_fn',
        'reward_label',
        'num_x0_samples',
    ]
    relevant_config = '_'.join([str(relevant_config[key]) for key in relevant_keys])

    return relevant_config


def load_texts(file):
    with open(file) as f:
        return [json.loads(line)["text"] for line in f]


def process_prompted_output(prompt_to_text, tokenizer, trim_len=50):
    prompt_to_data = {prompt: {} for prompt in prompt_to_text}

    for prompt, texts in prompt_to_text.items():
        cleaned_texts = []
        tokenized = []
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        prompt_to_data[prompt]["context_string"] = prompt
        prompt_to_data[prompt]["context_len"] = prompt_len
        prompt_to_data[prompt]["context"] = prompt_tokens

        for text in texts:
            tokenized_text = tokenizer.encode(text, add_special_tokens=False)[
                prompt_len : prompt_len + trim_len
            ]
            decoded_text = tokenizer.decode(tokenized_text)

            print('\t', decoded_text)

            cleaned_texts.append(decoded_text)
            tokenized.append(tokenized_text)

        prompt_to_data[prompt]["string"] = cleaned_texts
        prompt_to_data[prompt]["tokens"] = tokenized
        prompt_to_data[prompt]["len"] = len(tokenized[0])

    return prompt_to_data


def process_file(*, file, prompts, expected_per, tokenizer, max_len):
    config_info = file_to_exp_info(file)
    texts = load_texts(file)
    texts = [text.strip('<|endoftext|>') for text in texts]
    texts = ['\n\n' + text.strip() for text in texts]
    print(config_info)

    prompt_to_text = {prompt: [] for prompt in prompts}
    for text in texts:
        found_prompt = [prompt for prompt in prompts if text.startswith(prompt)]
        assert len(found_prompt) == 1
        found_prompt = found_prompt[0]
        prompt_to_text[found_prompt].append(text)

    # confirm that the number of samples per prompt is as expected
    for prompt, text in prompt_to_text.items():
        assert len(text) == expected_per

    prompt_to_data = process_prompted_output(prompt_to_text, tokenizer, max_len)
    return config_info, prompt_to_data


@click.command()
@click.option(
    '--glob_expression',
    default="../outputs/openwebtext-train/*/*/*/sample_evaluation/*/text_samples.jsonl",
    help='Glob pattern for input files.',
)
@click.option(
    '--prompt_path',
    default='pplm_discrim_prompts_orig.jsonl',
    help='Path to the prompt file.',
)
@click.option(
    '--max_len', default=50, type=int, help='Max length of generated text to consider.'
)
@click.option(
    '--expected_per',
    default=20,
    type=int,
    help='Expected number of samples per prompt.',
)
def main(glob_expression, prompt_path, max_len, expected_per):
    tokenizer_name = 'roberta-large'

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    prompts = get_possible_prompts(prompt_path)
    print(prompts)

    files = list(glob.glob(glob_expression))
    print(files)
    assert len(files) > 0

    for file in files:
        print(file)
        config_info, prompt_to_data = process_file(
            file=file,
            prompts=prompts,
            expected_per=expected_per,
            tokenizer=tokenizer,
            max_len=max_len,
        )
        # get parent dir path
        s_path = os.path.join(os.path.dirname(file), config_info + '_ssdlm_gen.jsonl')

        with open(s_path, 'w') as f:
            for _, data in prompt_to_data.items():
                f.write(json.dumps(data) + '\n')


if __name__ == '__main__':
    main()
