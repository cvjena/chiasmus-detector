from lxml import etree
import sys

fin = sys.argv[1]
fout = sys.argv[2]

tree = etree.parse(fin)
text = etree.tostring(tree, encoding='utf8', method='text').decode('utf8')
_ = text.split('\n')
text = ""
for l in _:
    line = l.strip()
    line = line.lstrip()
    text += line+"\n"

with open(fout, 'w') as f:
    f.write(text)
