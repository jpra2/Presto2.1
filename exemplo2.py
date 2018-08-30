import exemplo
import sys

conf = str(sys.argv[1])


with open(conf, 'r') as arq:
    for line in arq:
        print(line)
