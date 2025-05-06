path = "solution.py"
text = open(path, "r", encoding="utf-8").read()
fixed = text.replace("\u2011", "-")
open(path, "w", encoding="utf-8").write(fixed)
print("Done!")
