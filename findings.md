# Experiment Findings

## Overall Observation
- GPT-3.5-turbo significantly outperforms Bart-large on most of the tasks.
- Bart seems to be unable to generate sensible and coherent answers for any non-question-generation tasks, with or without fine-tuning.
- Bart does not seem to benefit much from few-shot examples, even after fine-tuning.
- For `subtask022_cosmosqa_passage_inappropriate_binary` specifically, Bart were not able to learn posterior distributions - it just predicts a one-way answer for all samples.
- In terms of negative examples, Bart seems to be unaffected due to its extremely limited ICL capability. GPT-3.5 performed slightly worse with negative examples.

## Rouge-L Scores
### GPT-3.5, 2 Positive Examples, 0 Negative Examples
```
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.5052912331319627
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.902
subtask033_winogrande_answer_generation
Rouge-L: 0.6163010156510128
subtask039_qasc_find_overlapping_words
Rouge-L: 0.5148075821389569
subtask040_qasc_question_generation
Rouge-L: 0.5699477473860566
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.6096463688199856
```
### GPT-3.5, 2 Positive Examples, 2 Negative Examples
```
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.5035106042201024
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.86
subtask033_winogrande_answer_generation
Rouge-L: 0.6107600055248259
subtask039_qasc_find_overlapping_words
Rouge-L: 0.512524370074368
subtask040_qasc_question_generation
Rouge-L: 0.5757572584030481
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.5548068802524304
```
### Bart, Full Prompt, No Few-Shot Examples
```
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.5411453369093194
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.064
subtask033_winogrande_answer_generation
Rouge-L: 0.11509468593372432
subtask039_qasc_find_overlapping_words
Rouge-L: 0.06787219417745105
subtask040_qasc_question_generation
Rouge-L: 0.384686696635509
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.18833830308348642
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.13210276078362063
```
### Bart, Full Prompt, 2 Positive Examples, 0 Negative Examples
```
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.5382647219561172
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.078
subtask033_winogrande_answer_generation
Rouge-L: 0.15954540265083617
subtask039_qasc_find_overlapping_words
Rouge-L: 0.029062790508557354
subtask040_qasc_question_generation
Rouge-L: 0.4150554977053583
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.14993429145529807
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.1571020958878323
```
### Bart, Full Prompt, 2 Positive Examples, 2 Negative Examples
```
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.5499692878819729
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.934
subtask033_winogrande_answer_generation
Rouge-L: 0.14034656274233237
subtask039_qasc_find_overlapping_words
Rouge-L: 0.05089441618928037
subtask040_qasc_question_generation
Rouge-L: 0.33198113671355867
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.0524577584903993
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.15396892442917315
```
### Bart, Full Prompt, 4 Positive Examples, 0 Negative Examples
```
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.5284733478777968
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.064
subtask033_winogrande_answer_generation
Rouge-L: 0.2133056449173975
subtask039_qasc_find_overlapping_words
Rouge-L: 0.03480777041222381
subtask040_qasc_question_generation
Rouge-L: 0.3797627424831536
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.07198313814494037
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.15605236929322513
```