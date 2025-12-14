MERGE_PROMPT = """Task: Merge a list of perspectives together to generate an abstract perspective.
Perspectives: 1. It is important to prioritize responsibilities as a parent. / 2. Prioritizing personal boundaries and belongings. / 3. Prioritize personal feelings and boundaries over familial obligations. / 4. Prioritize personal boundaries and independence. / 5. Respect for parental authority and boundaries. / 6. Prioritizing personal boundaries and individual parental responsibilities.
Answer: Prioritizing personal boundaries, independence, and responsibilities while respecting parental authority and familial obligations.

Task: Merge a list of perspectives together to generate an abstract perspective.
Perspectives: 1. Seek permission before bringing guests into someone else's home. / 2. It's wrong to expect others to conform to your personal expectations or standards, especially in matters as serious as pregnancy.
Answer: Respect others' boundaries and autonomy by seeking permission and refraining from imposing personal expectations or standards on them.

Task: Merge a list of perspectives together to generate an abstract perspective.
Perspectives: 1. Maintaining a positive and supportive relationship with loved ones is crucial. / 2. Building and maintaining relationships with loved ones is a valuable aspect of personal growth and connection.
Answer: Maintaining positive and supportive relationships with loved ones is essential for personal growth, connection, and overall well-being.

Task: Merge a list of perspectives together to generate an abstract perspective.
Perspectives: 
"""
DECOMPOSE_PROMPT = """Task: Resolve a list of conflicting perspectives and provide more context.
Perspectives: 1. It is important to prioritize responsibilities as a parent. / 2. As a parent, it is important to be free from burdens and responsibilities sometines.
Answer: Having responsibilities as a parent is expected when the parent is free from the burdens and personal difficulties.

Task:  Resolve a list of conflicting perspectives and provide more context.
Perspectives: 
"""
REMOVE_PREFIX="""Your task is to read a [Situation], and REMOVE some irrelevant items from [List of perspectives] when judging the acceptability of the situation. Answer 'REMOVE 1, 2, 3, .. ' for items to remove. Answer 'REMOVE NONE' if all items are relevant."""
ABSTRACT_PREFIX="""Your task is to read a [Situation] and MERGE similar items from [List of perspectives] when judging the acceptability of the situation. Answer 'MERGE 1, 2, 3, .. ' for items to combine. Answer 'MERGE NONE' if all items are distinct."""
DECOMPOSE_PREFIX="""Your task is to read a [Situation], and RESOLVE some conflicting items from [List of perspectives] when judging the acceptability of the situation. For instance, if item 1 is saying 'A is okay', and item 2 is saying 'A is not acceptable', then item 1 and 2 should be resolved. Answer as 'RESOLVE 1, 2, .. ' for items to resolve. Answer as 'RESOLVE NONE' if all perspectives are not conflicting to each other."""
ACTION_PREFIX="""Your task is to read a [Situation], and refine a given [List of perspectives] to better judge the acceptability of the situation. You need to choose one of four actions: REMOVE (when there are irrelevant perspectives to the situation), MERGE (when there are similar items that can be combined), RESOLVE (when there are conflicting items requiring more context), and END (when the perspectives are good enough to judge the situation). Choose only One action without any explanations:"""
INFERENCE_PREFIX="""Your will be given a [Situation] and a [List of perspectives]. Determine whether the situation is acceptable or not based on the given perspectives. Answer either 'Acceptable' or 'Unacceptable'."""

FIND_WINNING_VALUE_PROMPT = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
You will be given a [Situation] and a list of [Conflicting Values] observed in the situation. There's Person X who leaves a [Comment] and judges whether the situation is acceptable or not. Your task is to analyze Person X's [Comment], with respect to [Conflicting Values] in the situation. You need to accomplish two tasks in order. 
{Task 1} Determine which items in the [Conflicting Values] are mostly related to Person X's [Comment]. You may select only one of them, or select multiple items. Generate conflicting values observed in the comment ONLY IF none of the [Conflicting Values] are related to the [Comment].
{Task 2} For each of the selected [Conflicting Values] from {Task 1}, determine which value is more important to Person X. For instance, "A vs. B" is a chosen conflicting value and Person X thinks A is more important than B in the comment, the answer should be "A > B".

Your response should follow the format:
{
    'Conflicting Values': [
        "A vs. B",
        "C vs. D",
        ...
    ],
    'Winning Values': [
        "A > B",
        "D > C",
        ...
    ],
}

### Input:
__InputPlaceholder__

### Response:
"""

# VALUE_CLUSTER_CONCEPT_PROMPT = """
# I have this set of bullet point summaries of text examples:
# {examples}

# Please write a summary of {n_patterns} unifying patterns for these examples. For each high-level pattern, write a {n_words} word NAME for the pattern and an associated one-sentence ChatGPT PROMPT that could take in a new text example and determine whether the relevant pattern applies.
# Please also include {n_examples} example_ids for items that BEST exemplify the pattern. Please respond ONLY with a valid JSON in the following format :
# {{
#     "patterns": [
#         {{
#             "name": "<PATTERN_NAME_1>"
#             "prompt": "<PATTERN_PROMPT_1>"
#             "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]
#         }}
#         {{
#             "name": "<PATTERN_NAME_2>"
#             "prompt": "<PATTERN_PROMPT_2>"
#             "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]
#         }}
#     ]
# }}
# """
VALUE_CLUSTER_CONCEPT_PROMPT = """
I have this set of bullet point summaries of text examples:
{examples}

Please write a summary of unifying patterns for these examples. 
For each high-level pattern, write a {n_words} word NAME for the pattern and an associated one-sentence ChatGPT PROMPT that could take in a new text example and determine whether the relevant pattern applies.
Please also include {n_examples} example_ids for items that BEST exemplify the pattern. Please respond ONLY with a valid JSON in the following format :
{{
    "patterns": [
        {{
            "name": "<PATTERN_NAME_1>"
            "prompt": "<PATTERN_PROMPT_1>"
            "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]
        }}
        {{
            "name": "<PATTERN_NAME_2>"
            "prompt": "<PATTERN_PROMPT_2>"
            "example_ids": ["<EXAMPLE_ID_1>", "<EXAMPLE_ID_2>"]
        }}
    ]
}}
"""

ASSIGN_TO_CONCEPT_PROMPT = """
I have this set of text examples with their ids (e.g. ID#TEXT):
{examples}

I also have a set of themes generated by the text examples:
{patterns}

For each text example, assign them to ONE of the themes that matches the best.
Please respond ONLY with a valid JSON in the following format:
{{
    "assignment": [
        {{
            "TEXT_ID_1": "<THEME_NAME_1>",
            "TEXT_ID_2": "<THEME_NAME_2>",
            ..
        }}
    ]
}}
"""


REVIEW_REMOVE_PROMPT = """
I have this set of themes generated from text examples:
{concepts}

Please identify any themes that should be REMOVED because they are either:
(1) Too specific/narrow and would only describe a few examples, or 
(2) Too generic/broad and would describe nearly all examples.
If there no such themes, please leave the list empty.
Please respond ONLY with a valid JSON in the following format:

{{
    "remove": [ 
        "<THEME_NAME_5>",
        "<THEME_NAME_6>",
    ]
}}
"""

REVIEW_MERGE_PROMPT = """
I have this set of themes generated from text examples:
{concepts}

Please identify any PAIRS of themes that are similar or overlapping that should be MERGED together. 
Please respond ONLY with a valid JSON in the following format with the original themes and a new name and prompt for the merged theme. Do NOT simply combine the prior theme names or prompts, but come up with a new 4-5 word name and 1-sentence ChatGPT prompt. If there no similar themes, please leave the list empty.

{{
    "merge": [ 
        {{
            "original_themes": ["<THEME_NAME_A>", "<THEME_NAME_B>"],
            "merged_theme_name": "<THEME_NAME_AB>",
            "merged_theme_prompt": "<THEME_PROMPT_AB>",
        }},
        {{
            "original_themes": ["<THEME_NAME_C>", "<THEME_NAME_D>"],
            "merged_theme_name": "<THEME_NAME_CD>",
            "merged_theme_prompt": "<THEME_PROMPT_CD>",
        }}
    ]
}}
"""


REVIEW_SELECT_PROMPT = """
I have this set of themes generated from text examples:
{concepts}

Please select AT MOST {max_concepts} themes to include in the final set of themes. These themes should be the highest quality themes in the set: (1) NOT too generic or vague (should not describe most examples), (2) NOT too specific (should not only describe a small set of examples), and (3) NOT overlapping with other selected themes (they should capture a range of different patterns).
Please respond ONLY with a valid JSON in the following format:

{{
    "selected": [ 
        "<THEME_NAME_1>",
        "<THEME_NAME_2>",
    ]
}}
"""

GROUP_PROMPT = """
I have a JSON describing THEMES and their corresponding PROMPTS that could take in a new text example and determine whether the relevant theme applies:
{concepts}

Please group similar themes together so that AT MOST {max_concepts} themes left in the final set of themes. Each theme could be assigned to multiple theme groups, and should be assigned to at least one of the theme groups. These theme groups should be: (1) NOT too generic or vague (should not describe most examples), (2) NOT too specific (should not only describe a small set of examples), and (3) NOT overlapping with other selected themes (they should capture a range of different patterns).
Respond with ONLY a JSON with the following format, escaping any quotes within strings with a backslash:

{{
    "theme_groups": [
        {{
            "name": "<THEME_GROUP_NAME_1>",
            "prompt": "<THEME_GROUP_PROMPT_1>",
            "assigned_themes": ["<THEME_ID_A>", "<THEME_ID_B>, .."],
        }},
        {{
            "name": "<THEME_GROUP_NAME_2>",
            "prompt": "<THEME_GROUP_PROMPT_2>",
            "assigned_themes": ["<THEME_ID_C>", "<THEME_ID_D>, .."],
        }}
    ]
}}
"""

NAME_PROMPT = """
I have a JSON describing THEMES and their corresponding PROMPTS that could take in a new text example and determine whether the relevant theme applies:
{concepts}

Please write ONE summary of unifying pattern for these examples. 
For the high-level pattern, write a {n_words} word NAME for the pattern and an associated one-sentence ChatGPT PROMPT that could take in a new text example and determine whether the relevant pattern applies.
Please respond ONLY with a valid JSON in the following format:

{{
    "name": "<THEME_GROUP_NAME_1>",
    "prompt": "<THEME_GROUP_PROMPT_1>",
}}
"""

SCORE_PROMPT = """
CONTEXT: 
    I have the following text examples in a JSON:
    {examples_json}

    I also have a pattern named {concept_name} with the following PROMPT: 
    {concept_prompt}

TASK:
    For each example, please evaluate the PROMPT by providing a resulting ANSWER of ONE of the following multiple-choice options, including just the letter: 
    - A: Strongly agree
    - B: Agree
    - C: Neither agree nor disagree
    - D: Disagree
    - E: Strongly disagree
    Respond with ONLY a JSON with the following format, escaping any quotes within strings with a backslash:
    {{
        "pattern_results": [
            {{
                "example_id": "<example_id>",
                "answer": "<answer>",
            }}
        ]
    }}
"""

SCORE_PROMPT_ALT = """
CONTEXT: 
    I have the following text example:
    {example}

    I also have a set of themes: 
    {concepts}

TASK:
    For each theme, determine if the text example is generalized to the theme (i.e. whether the theme is generated by summarizing texts similar to the example), by providing a resulting ANSWER of ONE of the following multiple-choice options: 
    - A: Yes
    - B: Not determined
    - C: No
    ANSWER ONLY with comma-separated option letters. Note that the length of option letters should be same as the number of themes in the set, which is {n_concepts}.
"""


VALUE_TRADEOFF_BRIDGE_GEN = """
CONTEXT: 
###
    I have a SITUATION and a list of conflicting moral VALUES in the situation:
    {situation}
###
    I also have a set of Person X's history of comments that are left on other similar situations, followed by VALUE TRADE-OFFS observed in the comments: 
    {subjective_ground}
###
    Now given the initial situation, we observed that Person X choose {value_choice}.

TASK:
    Assume that the history of Person X's VALUE TRADE-OFFS can infer the Person X's value choice on the initial situation. Generate valid reasoning steps that would lead Person X's history of VALUE TRADE-OFFS to the final value choice. Reasoning should (1) Be strictly based on the history of VALUE TRADE-OFFS, (2) Logically make sense and conclude with the final value choice, and (3) Be described in 3 sentences.
"""

FIND_SUPPORT_PROMPT = """
### Instruction:
You will be given a [Situation] and there's Person X who leaves a [Comment] and provides [Judgment] whether the behavior of original poster of the situation is acceptable or not. You will also be given a list of [Person X's History] of past comments and judgments on similar situations. You need to accomplish two tasks in order. 
{Task 1} Determine which items in the [Person X's History] are mostly RELEVANT to Person X's [Comment] and [Judgment] on a given [Situation]. Note that "RELEVANT ITEMS" should be the supporting evidence you can use to infer Person X's [Judgment] on the [Situation]. You may select up to five items. If there is no relevant items in the [Person X's History], the output should be empty for this task.
{Task 2} Using the generated "RELEVANT ITEMS" from {Task 1}, generate short reasoning behind Person X's [Judgment] on the [Situation]. The reasoning should be STRICTLY BASED ON the items selected from the [Person X's History]. To generate a reasoning, try to understand the personality and subjectivity of Person X; consider what kind of moral values, social norms, biases Person X shows in the history. Reasoning should not exceed 5 sentences. If the output of {Task 1} is empty, this should be empty too.

Your input will be formatted as:
[[Situation]] text_description_of_situation
[[Person X's Comment]] personX_comment_on_situation
[[Person X's Judgment]] personX_judgment_on_situation
[[Person X's History]]
    [Item 1]
        (Situation) past_situation_description_1
        (Person X's Comment) personX_comment_on_past_situation_1
        (Person X's Judgment) personX_judgment_on_past_situation_1
    [Item 2]
        (Situation) past_situation_description_2
        (Person X's Comment) personX_comment_on_past_situation_2
        (Person X's Judgment) personX_judgment_on_past_situation_2
    ...

And your response should follow the format:
{
    "Relevant Items": [
        "[Item 1]",
        "[Item 3]",
        ...
    ],
    "Reasoning": "
        Person X would most likely judge the situation __Acceptable/Unacceptable__.
        Based on Person X's past history, ...
    "
}

### Input:
__InputPlaceholder__

### Response:
"""

FIND_SUPPORT_PROMPT_WITHVALUES = """
### Instruction:
You will be given a [Situation] and there's Person X who leaves a [Comment] and provides [Judgment] whether the behavior of original poster of the situation is acceptable or not. You will also be given a list of [Person X's History] of past comments and judgments on similar situations. You need to accomplish two tasks in order. 
{Task 1} Determine which items in the [Person X's History] are mostly RELEVANT to Person X's [Comment] and [Judgment] on a given [Situation]. Note that "RELEVANT ITEMS" should be the supporting evidence you can use to infer Person X's [Comment] and [Judgment] on the [Situation]. You may select up to five items. If there is no relevant items in the [Person X's History], the output should be empty for this task.
{Task 2} Using the generated "RELEVANT ITEMS" from {Task 1}, generate short reasoning behind Person X's [Judgment] on the [Situation]. The reasoning should be STRICTLY BASED ON the items selected from the [Person X's History]. To generate a reasoning, try to understand the personality and subjectivity of Person X; consider what kind of moral values, social norms, biases Person X shows in the history. Reasoning should not exceed 5 sentences. If the output of {Task 1} is empty, this should be empty too.

Your input will be formatted as:
[[Situation]] text_description_of_situation
[[Person X's Comment]] personX_comment_on_situation
[[Person X's Judgment]] personX_judgment_on_situation
[[Person X's History]]
    [Item 1]
        (Situation) past_situation_description_1
        (Person X's Comment) personX_comment_on_past_situation_1
        (Person X's Value) value_1 > value_2
        (Person X's Judgment) personX_judgment_on_past_situation_1
    [Item 2]
        (Situation) past_situation_description_2
        (Person X's Comment) personX_comment_on_past_situation_2
        (Person X's Value) value_3 > value_4
        (Person X's Judgment) personX_judgment_on_past_situation_2
    ...

And your response should follow the format:
{
    "Relevant Items": [
        "[Item 1]",
        "[Item 3]",
        ...
    ],
    "Reasoning": "
        Person X would most likely judge the situation __Acceptable/Unacceptable__.
        Based on Person X's past history, ...
    "
}

### Input:
__InputPlaceholder__

### Response:
"""