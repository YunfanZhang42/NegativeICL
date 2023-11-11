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
Rouge-L: 0.5048800870951279
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.902
subtask033_winogrande_answer_generation
Rouge-L: 0.6164151686548717
subtask039_qasc_find_overlapping_words
Rouge-L: 0.515208931239699
subtask040_qasc_question_generation
Rouge-L: 0.5695746894034718
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.607152800218064
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.42342633419049136
```
### GPT-3.5, 2 Positive Examples, 2 Negative Examples
```
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.5047654021921226
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.86
subtask033_winogrande_answer_generation
Rouge-L: 0.610898864537891
subtask039_qasc_find_overlapping_words
Rouge-L: 0.5128879176379156
subtask040_qasc_question_generation
Rouge-L: 0.5756901154458856
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.5528774577960986
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.39431892209242636
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
<<<<<<< HEAD:results/findings.md
Average Rouge-L: 0.2067434576670998
=======
```
### Bart, Full Prompt, On All Tasks, 0 Positive Examples, 0 Negative Examples
```
subtask002_quoref_answer_generation
Rouge-L: 0.38097793126414076
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.27723297276966063
subtask005_mctaco_wrong_answer_generation_event_duration
Rouge-L: 0.04903548002385212
subtask008_mctaco_wrong_answer_generation_transient_stationary
Rouge-L: 0.23102719565787744
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.064
subtask033_winogrande_answer_generation
Rouge-L: 0.19712258087876877
subtask034_winogrande_question_modification_object
Rouge-L: 0.5196302905818251
subtask039_qasc_find_overlapping_words
Rouge-L: 0.0770045415768108
subtask040_qasc_question_generation
Rouge-L: 0.356492981665816
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.043693778399310185
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.13441370171322511
subtask052_multirc_identify_bad_question
Rouge-L: 0.08974358974358974
```
### Bart, Full Prompt, On All Tasks, As Many Positive Examples As Possible, 0 Negative Examples
```
subtask002_quoref_answer_generation
Rouge-L: 0.25765152696437915
subtask003_mctaco_question_generation_event_duration
Rouge-L: 0.4477394884964331
subtask005_mctaco_wrong_answer_generation_event_duration
Rouge-L: 0.061683206973904635
subtask008_mctaco_wrong_answer_generation_transient_stationary
Rouge-L: 0.1852979879968516
subtask022_cosmosqa_passage_inappropriate_binary
Rouge-L: 0.064
subtask033_winogrande_answer_generation
Rouge-L: 0.24563067653938014
subtask034_winogrande_question_modification_object
Rouge-L: 0.45937026644894763
subtask039_qasc_find_overlapping_words
Rouge-L: 0.1422994823552512
subtask040_qasc_question_generation
Rouge-L: 0.47272604226395154
subtask044_essential_terms_identifying_essential_words
Rouge-L: 0.00846556010765126
subtask045_miscellaneous_sentence_paraphrasing
Rouge-L: 0.07481759882562085
subtask052_multirc_identify_bad_question
Rouge-L: 0.2532051282051282
>>>>>>> c468341774494e90121f433e30d972298e0a2bf8:findings.md
```
