import re

with open("demo.txt", "r", encoding="utf-8") as f:
    text = f.read()

text = re.sub(r'\.\s*', '.\n', text)

with open("demo.txt", "w", encoding="utf-8") as f:
    f.write(text)

print("Done! New line added after every period.")
