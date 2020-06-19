# 데이터과학 group 2
# 하이퍼파라미터 지정 및 하이퍼파라미터 기록 저장

import torch, os

class hparams():

	def __init__(self):
		self.hidden_size = 200
		self.learning_rate = 0.01
		self.n_iters = 150000
		self.dropout = 0.1

		self.save_epoch = 5000
		self.max_len = 18 # 임의로 저장한 값. 
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	def save_log(self, dir_) :
		save_dir = os.path.join(dir_,"save_log.txt")
		with open(save_dir,"w") as f:
			
			txt = """
hidden_size: {}
learning_rate: {}
iterations: {}
dropout: {}
			""".format(self.hidden_size,self.learning_rate,self.n_iters,self.dropout)
			
			f.write(txt)



