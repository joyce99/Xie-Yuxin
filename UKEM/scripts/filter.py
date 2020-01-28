from __future__ import print_function
from extractTrain import search, myfile
import logging
import os.path
import sys
import re
import codecs

#reload(sys)
#sys.setdefaultencoding('utf8')
def main1():
	folder = r"/Users/mac/Documents/LearnPython/KeywordExtraction/data/SemEval2010/train"
	filters = ['C','H','I','J']
	# filters = ['C']
	allfile = []
	allfile = search(folder, filters, allfile)
	file_len = len(allfile)
	print('共查找到%d个摘要文件' % (file_len))
	for f in allfile:
		cur_name = os.path.basename(f)
		letter = os.path.basename(f)[0]
		for i in range(2, len(os.path.basename(f))):
			if os.path.basename(f)[i] != '.':
				continue
			else:
				number = int(os.path.basename(f)[2:i])
				break
		output_file = open('../data/SemEval2010/train_removed/%s-%d.txt' % (letter, number), 'w', encoding='utf-8')
		with open(f, 'r', encoding='utf-8') as curf:
			for line in curf.readlines():
				ss = re.sub(
					"[\s+\.\!\/_,;\[\]><•¿#&«»∗`={}|1234567890¡?():$%^*(+\"\']+|[+！，。？；：、【】《》“”‘’~@#￥%……&*（）''""]+", " ", line)
				ss += "\n"
				output_file.write("".join(ss.lower()))

		output_file.close()


def main2():
	program = os.path.basename(sys.argv[0])
	logger = logging.getLogger(program)
	logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
	logging.root.setLevel(level=logging.INFO)
	logger.info("running %s" % ' '.join(sys.argv))
	if len(sys.argv) != 3:
		print("Using: python filter.py xxx.txt xxxx.txt")
		sys.exit(1)
	inp, outp = sys.argv[1:3]
	output = codecs.open(outp, 'w', encoding='utf-8')
	inp = codecs.open(inp, 'r', encoding='utf-8')
	i = 0
	for line in inp.readlines():
		# ss = re.sub("[\s+\.\!\/_,;-><¿#&«-»=|1234567890¡?():$%^*(+\"\']+|[+——！，。？；：、【】《》“”‘’~@#￥%……&*（）''""]+".decode("utf-8"), " ".decode("utf-8"),line)
		# ss = re.sub("[\s+\.\!\/_,;\[\]><•¿#&«»∗`{}=|1234567890¡?():$%^*(+\"\']+|[+！，。？℃；：、【】《》“”‘’~@#￥%……&*（）''""]+", " ",
		# 			line)
		ss = re.sub('[。，：；、1234567890（）]+', '', line)
		# ss += "\n"
		# output.write("".join(ss.lower()))
		output.write(ss)
		i = i + 1
		if (i % 10000 == 0):
			logger.info("Saved " + str(i) + " articles")
	# break
	output.close()
	inp.close()
	logger.info("Finished removed words!")

def my_write(begin_write, cur_line, output_name):
		if begin_write:
			out_file = open(output_name, 'a', encoding='utf-8')
			out_file.write(cur_line)
			out_file.close()

def get_content():
	inp = open('../data/raw/SemEval2010_train_raw.txt', 'r', encoding='utf-8')
	begin_write = False
	lines = inp.readlines()
	for line in lines:
		if 'ABSTRACT' in line:
			begin_write = True
			continue
		if 'REFERENCES' in line:
			begin_write = False
			continue
		my_write(begin_write, line, '../data/SE2010_content.txt')
	inp.close()


if __name__ == '__main__':
	main2()
	#get_content()
	# f1 = open('../data/raw/SemEval2010_train_raw.txt', 'r', encoding='utf-8')
	# f2 = open('../data/SE2010_content.txt', 'r', encoding='utf-8')
	# print(len(f1.readlines()))
	# print(len(f2.readlines()))
	# f1.close()
	# f2.close()