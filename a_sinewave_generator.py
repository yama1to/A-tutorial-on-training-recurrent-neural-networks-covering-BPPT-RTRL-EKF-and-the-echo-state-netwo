import numpy as np
import matplotlib.pyplot as plt

class sinewave_generator:
	def __init__(self,x_nodes_num,y_nodes_num,learning_times,f = np.tanh):

		#reservoir層の重みを記録
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
		M = self.x[101:]
		#print(M.shape)
		T = np.array([[self.d(n)] for n in range(101,len(self.x))])
		w_out = np.linalg.pinv(M) @ T
		self.w_out = w_out

	#reservoir層の次状態を取得
	def next_x(self,y):
		
		a = self.w @ np.array([self.x[-1]]).T
		b = self.w_back.T @ y
		x_next_state = self.f(a+b).T
		
		#x_next_state = self.f(self.w_back.T @ y).T
		return x_next_state *0.2 #(1,20)

	#訓練
	def train(self,learning_times):
		for i in range(learning_times):
			if (i+1) % 100 == 0:
				print(str(i+1)+"回目")
			#y = np.sum(self.w_out @ np.array([self.x[-1]]))
			y = self.d(i)
			#print(y)
			y = np.array([[y]])
			#print(self.x.shape)
			self.x = np.append(self.x,self.next_x(y),axis = 0)
			#print(self.x.shape,self.next_x(y).shape)
			#print(i)

		self.update_w_out()

	#予測
	def predict(self,learning_times,predict_times):
		predicted_outputs = np.array(self.d(learning_times-1)) #最後の教師
		
		#print(x)
		for _ in range(predict_times):
			a = self.w_out.T * np.array([self.x[-1]])
			#print(a.shape,self.w_out.shape,np.array([self.x[-1]]).shape)
			y = np.sum(a)
			y = np.array([[y]])
			self.x = np.append(self.x,self.next_x(y),axis = 0)
			predicted_outputs = np.append(predicted_outputs,y)
		return predicted_outputs

	#誤差計算 mean squared training error
	def MSE(self,start,end,predicted_outputs):
		mse = 0
		for n in range(start,end+1):
			diff = self.d(n-1) - predicted_outputs[n-start]
			#print(diff)
			#print(diff)
			mse += (diff)**2
			#print(n-start)
		mse = mse/200
		return mse

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

		#xの中身
		for i in range(1,21):
			ax = fig.add_subplot(5,5,i) #縦横5x5個グラフをつくる。それのi番目
			y = self.x[:,i-1]
			ax.plot(sequence,y[sequence[0]:sequence[-1]+1])

		#教師信号
		ax = fig.add_subplot(5,5,21)
		y = list(self.d(n) for n in sequence)
		ax.plot(sequence,y)
		plt.show()
		#print(self.x.shape)

	def p_plot(self,learning_times,predict_times,predict):
		x_test = list(range(learning_times,learning_times+predict_times+1))
		x_train = range(1,learning_times)
		y_train = list(self.d(n) for n in range(1,learning_times))

		plt.plot(x_train,y_train)
		plt.plot(x_test,predict)
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

	model.train(learning_times)#train
	predict = model.predict(learning_times,predict_times)#predict
	mse = model.MSE(learning_times,learning_times+predict_times,predict)
	print("教師と予測の平均２乗誤差は ["+str(mse)+"]")#目標は 5.6e-12の誤差

	
	#model.p_plot(learning_times,predict_times,predict)

	#r = list(range(101,151))
	#model.x_plot(r)

	return mse



main()


mse = []
for i in range(300):
	if main() < 5.6e-8:
		mse.append(main())

mse = sorted(mse)
print(len(mse))






