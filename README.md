# ATNLP Coursework

See README.md files in each directory, partI/ and partII/, and coursework main PDF under coursework.pdf.

Gradescope form to submit answers will only be open when the assignment is officially released.

The file coursework-all.zip consists of all relevant content. You can download it to solve the assignment locally.

Short history:

Update 7/2/2026: The file partI/shapley_value_evaluation.py was changed with two additional comments (no actual change in code); The file partI/utils.py was changed (openai as argument to a function changed to openai_m, for clarity/clean code)
		 As a result, coursework-all.zip was updated as well

Update 9/2/2026: Changes requirements.txt in partII/, as the default transformers package vesion on Google colab has changed, and was no longer compatible with the peft package. The correct transformers package is now enforced in requirements.txt.

Update 15/2/2026: Minor change into one of the commands provided in partII (running main.py). Last backslash was removed, as unnecessary. Only the coursework.pdf was changed.

Update 21/2/2026: Change to partII/evaluation/main.py so we can evaluate with GRPO (Q4) with both GRPO adapter and the SFT adapter (model merging with more than one model).

Update 22/2/2026: Minor change to partII/evaluation/main.py (bfloat -> float).
