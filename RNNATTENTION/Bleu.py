# Calculating Bleu Score

from math import exp

class Bleu():

	def __init__(self, output_sentence, target_sentence) :
		self.output = output_sentence
		self.target = target_sentence

		# <EOS tagging 지우기>
		if "<EOS>" in self.output :
			del self.output[-1]
		
	def one_gram(self) :
		corresponds = 0
		denominator = len(self.output)

		for word in self.output :
			if word in self.target :
				corresponds += 1

		return corresponds/denominator

	def two_gram(self) :
		corresponds = 0
		denominator = len(self.output) - 1

		t = self.target
		o = self.output
		target = [(t[num],t[num+1]) for num in range(len(t)-1)]
		output = [(o[num],o[num+1]) for num in range(len(o)-1)]

		for words in output : 
			if words in target :
				corresponds += 1 

		return corresponds/denominator

	def three_gram(self) :
		corresponds = 0
		denominator = len(self.output) - 2

		t = self.target
		o = self.output
		target = [(t[num],t[num+1],t[num+2]) for num in range(len(t)-2)]
		output = [(o[num],o[num+1],o[num+2]) for num in range(len(o)-2)]

		for words in output : 
			if words in target :
				corresponds += 1 

		return corresponds/denominator

	def four_gram(self) :
		corresponds = 0
		denominator = len(self.output) - 3

		t = self.target
		o = self.output
		target = [(t[num],t[num+1],t[num+2],t[num+3]) for num in range(len(t)-3)]
		output = [(o[num],o[num+1],o[num+2],o[num+3]) for num in range(len(o)-3)]

		for words in output : 
			if words in target :
				corresponds += 1 

		return corresponds/denominator

	# Penalty for too short sentences
	def brevity_penalty(self) :
		o = len(self.output)
		t = len(self.target)
		if o >= t :
			return 1
		else :
			return exp(1-t/o)

	def scoring(self) :
		b_p = Bleu.brevity_penalty(self)

		if len(self.output) == 1 :
			o_g = Bleu.one_gram(self)
			return b_p*min(1,len(self.output)/len(self.target))*o_g

		elif len(self.output) == 2 :
			o_g = Bleu.one_gram(self)
			tw_g = Bleu.two_gram(self)
			return b_p*min(1,len(self.output)/len(self.target))*pow(o_g*tw_g,1/2)

		elif len(self.output) == 3 :
			o_g = Bleu.one_gram(self)
			tw_g = Bleu.two_gram(self)
			th_g = Bleu.three_gram(self)
			return b_p*min(1,len(self.output)/len(self.target))*pow(o_g*tw_g*th_g,1/3)

		else :
			o_g = Bleu.one_gram(self)
			tw_g = Bleu.two_gram(self)
			th_g = Bleu.three_gram(self)
			f_g = Bleu.four_gram(self)
			return b_p*min(1,len(self.output)/len(self.target))*pow(o_g*tw_g*th_g*f_g,1/4)

if __name__ == "__main__" :
	output = ["I","am","very","happy","right","now","."]
	target = ["I","am","very","happy","hello","now","."]

	bleu = Bleu(output,target)
	print(bleu.scoring())
