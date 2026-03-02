'''
Question 3
Create a variable `INSTRUCTION` and fill in the prompt with instructions provided in the file `prompt-instruction.txt.`
'''
#raise NotImplementedError("Create a variable `INSTRUCTION` and fill in the prompt with instructions provided in the file `prompt-instruction.txt.`")   
with open("partI/MMLU-Redux-college_mathematics_prompts/comat.txt", "r", encoding="utf-8") as f:
    INSTRUCTION = f.read().strip()