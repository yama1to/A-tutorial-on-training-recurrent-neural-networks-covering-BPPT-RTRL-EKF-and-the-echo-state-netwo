import numpy as np
import matplotlib.pyplot as plt


"""
input 0  
DR 20 ⇨x
out 1 ⇨y

W_out x→y
W_back y→x
W x→x

M
T
"""

class sinewave_generator:
	def __init__(self,reservoir_nodes,output_nodes,learning_times,activator = np.tanh):

		#reservoir層の重みを記録　最初はゼロ行列
		#self.log_reservoir_nodes = np.array([np.zeros(reservoir_nodes)])
		self.log_reservoir_nodes = np.array([np.random.normal(0,1,reservoir_nodes)])
		#重み 
		self.reservoir_weight = self.standard_init_weight(reservoir_nodes,reservoir_nodes)
		self.back_weight = self.standard_init_weight(reservoir_nodes,reservoir_nodes)
			#更新されるのはoutput_weightだけ。
		self.output_weight = self.standard_init_weight(reservoir_nodes,output_nodes)

		#ノード数
		self.reservoir_nodes = reservoir_nodes
		self.output_nodes = output_nodes

		#活性化関数
		self.activator = activator

	#出力層の重みの更新
	def update_weight_output(self):
		M = self.log_reservoir_nodes
		T = [self.d(n) for n in range(1,len(M)+1)]
		#print(M)
		output_weight = np.linalg.pinv(M) @ T
		self.output_weight = output_weight

	#reservoir層の次状態を取得
	def next_state_reservoir(self,current_state):
		next_state = self.reservoir_weight @ self.log_reservoir_nodes[-1]
		#print(next_state)
		#print(self.back_weight.shape,np.array([current_state]).shape)
		next_state += self.back_weight @ current_state
		#print(self.log_reservoir_nodes[-1],current_state)
		next_state = self.activator(next_state)*0.1
		return next_state

	#訓練
	def train(self,learning_times):
		s = 1
		for i in range(learning_times):
			if i/100>s:
				print(str(i-1)+"回目")
				s += 1
			current_state = np.array(self.log_reservoir_nodes[-1])
			#current_state = np.array([self.d(i) for _ in range(20)])
			self.log_reservoir_nodes = np.append(self.log_reservoir_nodes,
				[self.next_state_reservoir(current_state)],axis = 0)
			#print(self.log_reservoir_nodes.shape)
			self.update_weight_output()

	#予測
	def predict(self,learning_times,predict_times):
		predicted_outputs = [self.d(learning_times)]
		reservoir = self.log_reservoir_nodes[-1]

		for _ in range(predict_times):
			reservoir = self.next_state_reservoir(reservoir)
			predicted_outputs.append(reservoir @ self.output_weight)
		return predicted_outputs

	#誤差計算 mean squared training error
	def MSE(self,start,end,predicted_outputs):
		mse = 0
		#print(start,end)
		for n in range(start+1,end):
			diff = self.d(n) - predicted_outputs[n-start-1]
			#print(diff)
			mse += (diff)**2
		mse = mse/200
		print("教師と予測の平均２乗誤差は ["+str(mse)+"]")

	#教師信号
	def d(self,n):
		return np.sin(n/4)/2

	#初期値
	def standard_init_weight(self,c,v):
		w = np.random.normal(0,1,c*v).reshape([c,v])
		return w
	#初期値
	def zeros_init_weight(self,c,v):
		return np.zeros([c,v])

	#reservior層の中身を見る。０に収束させたい。
	def reservior_plot(self,sequence):
		fig = plt.figure()
		ax = []
		for i in range(1,21):
			ax = fig.add_subplot(4,5,i)
			y = self.log_reservoir_nodes[:,i-1]

			#print(y.shape)
			ax.plot(sequence,y[:sequence[-1]])
		plt.show()

def main():

	reservoir_nodes = 20
	output_nodes = 1
	learning_times = 300 #default 300
	predict_times = 50



	model = sinewave_generator(
		reservoir_nodes = reservoir_nodes,
		output_nodes = output_nodes,
		learning_times = learning_times)

	model.train(learning_times)

	predict = model.predict(learning_times,predict_times)
	print(predict)
	x_test = list(range(learning_times,learning_times+predict_times+1))
	x_train = range(1,learning_times)
	y_train = list(model.d(n) for n in range(1,learning_times))
	
	model.MSE(x_test[0],x_test[-1],predict)
	
	plt.plot(x_train,y_train)
	plt.plot(x_test,predict)
	plt.show()
	

	model.reservior_plot(list(range(1,predict_times+1)))

	#print(np.array([np.zeros(reservoir_nodes)]))
	#print(np.array([np.random.normal(0,1,reservoir_nodes)]))

main()





