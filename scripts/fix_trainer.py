with open("src/ai/trainer.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "def run(self):" in line:
        pass # just to find it
    if "return True" in line and i > 230:
        lines[i] = "        return True\n"
        
with open("src/ai/trainer.py", "w", encoding="utf-8") as f:
    f.writelines(lines)
