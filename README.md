**Status**: Archive (code is provided as-is, no updates expected)

# Grade School Math

#### [[Blog Post]](https://openai.com/blog/grade-school-math/) [[Paper]](https://arxiv.org/abs/2110.14168)

State-of-the-art language models can match human performance on many tasks, but they still struggle to robustly perform multi-step mathematical reasoning. To diagnose the failures of current models and support research, we're releasing GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math word problems. We find that even the largest transformer models fail to achieve high test performance, despite the conceptual simplicity of this problem distribution.

最新的语言模型在许多任务上的表现已经能够与人类相媲美，但在进行多步数学推理方面仍然存在困难。为了诊断当前模型的不足并支持相关研究，我们发布了GSM8K数据集。这是一个包含8500个高质量、语言学多样化的数学应用题的数据集，这些题目适合小学生水平。然而，即使是最强大的Transformer模型，在这种问题分布上也难以取得较高的测试性能，尽管这些问题从概念上来说相对简单。

<p align="center">
    <img src="grade_school_math/img/example_problems.png" height="300"/>
</p>

## Dataset Details

GSM8K consists of 8.5K high quality grade school math problems created by human problem writers. We segmented these into 7.5K training problems and 1K test problems. These problems take between 2 and 8 steps to solve, and solutions primarily involve performing a sequence of elementary calculations using basic arithmetic operations (+ - / \*) to reach the final answer. A bright middle school student should be able to solve every problem.

GSM8K由8500个高质量的小学数学问题组成，这些问题均由人类编写者创作。我们将这些问题划分为7500个训练问题和1000个测试问题。解决这些问题需要2到8个步骤，解决方案主要涉及使用基本算术运算（加、减、乘、除）进行一系列初等计算以得出最终答案。一个聪明的中学生应该能够解决所有这些问题。

The raw data files can be found in:

- `grade_school_math/data/train.jsonl`
- `grade_school_math/data/test.jsonl`

Each line of those files corresponds to a single grade school math problem, saved as a json dictionary (with a "question" key and an "answer" key). The answer is formatted such that it uses calculation annotations and so that the final numeric solution is the final line of the solution, preceded by `####`.

这些文件的每一行都对应一个小学数学问题，以JSON字典的形式保存（包含“question”键和“answer”键）。答案的格式使用了计算注释，并且最终的数值解是解决方案的最后一行，前面有####标记。

### Calculation Annotations

Our models frequently fail to accurately perform calculations. Although larger models make fewer arithmetic mistakes than smaller models, this remains a common source of errors. To mitigate this issue, we train our models to use a calculator by injecting calculation annotations into the training set. At training time, we simply finetune on this language data as is. At test time, a calculator will override sampling when the model chooses to use these annotations. An example implementation of the calculator sampling can be found in `calculator.py`.

我们的模型常常无法准确地进行计算。尽管较大的模型比小模型犯的算术错误更少，但计算错误仍然是一个常见的问题来源。为了解决这一问题，我们在训练集中加入计算注释，训练模型使用计算器。在训练阶段，我们直接对这些语言数据进行微调。在测试阶段，当模型选择使用这些注释时，计算器将取代采样操作。计算器采样的一个示例实现可以在calculator.py中找到。

If you would like to remove the calculator annotations, simply remove any string that starts with `<<` and ends with `>>`.

### Solution Extracting

To extract the final numeric solution for a particular question, simply parse the completion to extract the numeric value immediately following the `####` token. Some example python code to do so is shown in `dataset.py:is_correct`.

为了提取某个特定问题的最终数值解决方案，只需解析完成部分以提取紧跟在####标记之后的数值。在dataset.py:is_correct中展示了执行此操作的示例Python代码。

### Socratic Dataset

During our research, we also investigated a modified solution format that injects automatically generated "Socratic subquestions" before each step. Although we ultimately did not use this format for any experiments in the paper, we make this data available to anyone who is interested.

在我们的研究中，我们还研究了一种修改后的解决方案格式，该格式在每个步骤之前插入自动生成的“苏格拉底式子问题”。尽管我们最终没有在论文的任何实验中使用这种格式，但我们将其数据提供给任何感兴趣的人员。

We show an example below, with the socratic subquestions in bold:

<pre>
A carnival snack booth made $50 selling popcorn each day. It made three times as much selling cotton candy. For a 5-day activity, the booth has to pay $30 rent and $75 for the cost of the ingredients. How much did the booth earn for 5 days after paying the rent and the cost of ingredients?
<b>How much did the booth make selling cotton candy each day? **</b> The booth made $50 x 3 = $<<50*3=150>>150 selling cotton candy each day.
<b>How much did the booth make in a day? **</b> In a day, the booth made a total of $150 + $50 = $<<150+50=200>>200.
<b>How much did the booth make in 5 days? **</b> In 5 days, they made a total of $200 x 5 = $<<200*5=1000>>1000.
<b>How much did the booth have to pay? **</b> The booth has to pay a total of $30 + $75 = $<<30+75=105>>105.
<b>How much did the booth earn after paying the rent and the cost of ingredients? **</b> Thus, the booth earned $1000 - $105 = $<<1000-105=895>>895.
</pre>

We generated each Socratic subquestion by conditioning on each ground truth (contractor-provided) step in a solution, using a model specifically finetuned for this task (on around 800 examples). To construct the full Socratic dataset, each step in the solution was prefixed by the model-generated Socratic subquestion. Steps were otherwise left untouched.

我们通过使用一个专门为这项任务微调的模型（基于大约800个示例），针对解决方案中的每一步（由承包商提供的真实步骤）生成每个苏格拉底式子问题。为了构建完整的苏格拉底数据集，我们将模型生成的苏格拉底式子问题作为解决方案中每一步的前缀。其他方面，步骤保持不变。

These data files can be found in:

- `grade_school_math/data/train_socratic.jsonl`
- `grade_school_math/data/test_socratic.jsonl`

## View Model Solutions

For each test question, we provide solutions generated from 6B finetuning, 6B verification, 175B finetuning and 175B verification. This data can be found in:

- `grade_school_math/data/example_model_solutions.jsonl`

To view these results problem-by-problem, run:

```bash
python view_model_solutions.py
```

Note: These model-generated samples used a slightly older version of the calculator. Previous implementation bugs led to calculator failures in roughly 1% of model samples. Those issues have been fixed in the codebase, but since the samples have not been regenerated, occasional calculation errors are present.

注意：这些由模型生成的样本使用了一个稍微旧一点版本的计算器。由于之前的实现错误，大约有1%的模型样本在使用计算器时出现了故障。这些问题已经在代码库中修复了，但由于样本尚未重新生成，因此偶尔仍会出现计算错误。

## Citation

Please use the below BibTeX entry to cite this dataset:

```
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```

# Usage

We present a basic example of training a GPT2 sized model and using the calculator in the sampling process. We include this code for illustrative purposes only. This pipeline was not used for any experiments in the paper.

我们展示了一个训练与GPT-2规模相当的模型，并在采样过程中使用计算器的基本示例。我们仅出于说明目的包含此代码。这一流程并未用于论文中的任何实验。

**Training a Model**

```bash
python train.py
```

**Sampling from the Model**

```bash
python sample.py
```

The core calculator sampling logic can be found in `calculator.py:sample`. Note that this code is inefficient as implemented. Specifically, the function does not support batches, and does not cache activations from previous tokens.

核心的计算器采样逻辑可以在calculator.py:sample中找到。请注意，这段代码的实现是低效的。具体来说，该函数不支持批处理，也不缓存以前的令牌的激活。
