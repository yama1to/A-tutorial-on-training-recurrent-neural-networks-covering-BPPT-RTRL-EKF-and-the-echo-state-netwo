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
	def __init__(self,x_nodes_num,y_nodes_num,learning_times,f = np.tanh):

		#reservoir層の重みを記録
		#self.x = np.array([np.zeros(x_nodes_num)])
		self.x = self.standard_init_weight(y_nodes_num,x_nodes_num)
		
		#重み 
		self.w = self.standard_init_weight(x_nodes_num,x_nodes_num)
		self.w_back = self.standard_init_weight(y_nodes_num,x_nodes_num)
		#更新されるのはw_outだけ。
		self.w_out = self.standard_init_weight(x_nodes_num,y_nodes_num)



		#ノード数
		self.x_nodes_num = x_nodes_num
		self.y_nodes_num = y_nodes_num

		#活性化関数
		self.f = f

	

	#出力層の重みの更新
	def update_w_out(self):
		M = self.x
		T = np.array([[self.d(n)] for n in range(1,len(M)+1)])
		#print(M,T)
		w_out = np.linalg.pinv(M) @ T
		#print(w_out,self.w_out)
		self.w_out = w_out

	#reservoir層の次状態を取得
	def next_x(self,y):
		x_next_state = self.f(self.w @ self.x[-1] + self.w_back.T @ y)
		return x_next_state 

	#訓練
	def train(self,learning_times):
		s = 1
		
		for i in range(learning_times):
			if (i+1) % 100 == 0:
				print(str(i+1)+"回目")
			y = np.sum(self.w_out @ np.array([self.x[-1]]))
			y = np.array([[y]])
			self.x = np.append(self.x,self.next_x(y),axis = 0)
			if i>=100:
				self.update_w_out()

	#予測
	def predict(self,learning_times,predict_times):
		predicted_outputs = np.array(self.d(learning_times))
		
		#print(x)
		for _ in range(predict_times):
			#print(self.w_out.shape,array([self.x[-1].shape)
			y = np.sum(self.w_out @ np.array([self.x[-1]]))
			y = np.array([[y]])
			self.x = np.append(self.x,self.next_x(y),axis = 0)
			predicted_outputs = np.append(predicted_outputs,y)
		return predicted_outputs

	#誤差計算 mean squared training error
	def MSE(self,start,end,predicted_outputs):
		mse = 0
		for n in range(start+1,end):
			diff = self.d(n) - predicted_outputs[n-start-1]
			#print(diff)
			mse += (diff)**2
		mse = mse/200
		print("教師と予測の平均２乗誤差は ["+str(mse)+"]")#目標は 1.2e-13の誤差

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
	def x_plot(self,sequence):
		fig = plt.figure()
		for i in range(1,21):
			ax = fig.add_subplot(4,5,i)
			y = self.x[:,i-1]
			ax.plot(sequence,y[:sequence[-1]])
		plt.show()

def main():

	x_nodes_num = 20
	y_nodes_num = 1
	learning_times = 300 #default 300
	predict_times = 50



	model = sinewave_generator(
		x_nodes_num = x_nodes_num,
		y_nodes_num = y_nodes_num,
		learning_times = learning_times)

	model.train(learning_times)

	predict = model.predict(learning_times,predict_times)
	#print(predict)
	x_test = list(range(learning_times,learning_times+predict_times+1))
	x_train = range(1,learning_times)
	y_train = list(model.d(n) for n in range(1,learning_times))
	
	model.MSE(x_test[0],x_test[-1],predict)
	
	plt.plot(x_train,y_train)
	plt.plot(x_test,predict)
	plt.show()
	

	model.x_plot(list(range(1,predict_times+1)))

	#print(np.array([np.zeros(x_nodes_num)]))
	#print(np.array([np.random.normal(0,1,x_nodes_num)]))

main()





