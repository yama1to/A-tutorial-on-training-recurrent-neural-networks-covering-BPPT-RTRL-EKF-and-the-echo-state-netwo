import numpy as np 
import matplotlib.pyplot as plt 
import math

class reservoir_billiard_system:
	def __init__(self):
		self.u_num = 2		#input 
		self.r_num = 100	#reservoir
		self.y_num = 2		#output

		
		self.u = np.zeros((60000,self.u_num))#(60000,u_num)
		self.u_s = np.array([])#(60000,u_num)
		
		self.r_x = np.random.uniform(0,1,self.r_num)
		self.r_s = np.zeros((self.r_num))

		self.prev_r_s = np.zeros((self.r_num))
		self.prev_r_x = np.zeros((self.r_num))
		

		self.w_in = np.zeros((self.r_num,self.u_num))
		self.w_rec = np.zeros((self.r_num,self.r_num))
		self.w_out = np.zeros((self.r_num,self.y_num))

		self.temperature = 1	#temperature
		self.f = np.tanh 		#activation function
		self.fy = np.tanh 	#activation function

		self.alpha_r = 0.25		#connectivity strength
		self.alpha_i = 0.2		#
		self.alpha_s = 0.6		#

		self.beta_r = 0.1		#connectivity rate
		self.beta_i = 0.1		#

		self.t_div = 200	#仮想的に連続値を生み出すため、timeが1あたり、200個にさらに時間を分ける。つまり0.05刻み
		self.test_time = 300 #300回
		self.time = np.round(np.arange(0,self.test_time,1/self.t_div),3) #0~300を0.005刻みで進めていく。実質60000
		self.T0 = 50

		self.lambda0 = 0.1 #regurarization parameter
		self.E = np.identity(self.test_time - self.T0)	#identity matrix


		self.R = np.zeros((self.test_time - self.T0 , self.r_num))


		self.r_x_collect = np.zeros((self.test_time*self.t_div,self.r_num))
		self.r_s_collect = np.zeros((self.test_time*self.t_div,self.r_num))
		self.J_collect = np.zeros((self.test_time*self.t_div,self.r_num))
		#self.r_s = np.zeros((self.test_time*self.t_div,self.r_num))

	def u_init_generater(self,):
		cu = np.linspace(0,1,self.u_num)
		u1 = np.sin(self.time/8)*0.6
		u2 = np.sin(1+(self.time/8))*0.6
		self.u = self.f(np.append(u1,u2).reshape((2,self.test_time*self.t_div)))

	def w_rec_setting(self,):
		w_rec_0 = np.zeros((self.r_num,self.r_num))
		ran = (self.beta_r * self.r_num**2)
		half = int(ran/2)
		w_rec_0[:half] = 1
		w_rec_0[half:int(ran)] = -1
		np.random.shuffle(w_rec_0)
		value , _ = np.linalg.eig(w_rec_0)#　固有値
		w_rec = self.alpha_r * w_rec_0 / max(abs(value))
		self.w_rec = w_rec #(r_num,r_num)

	def w_in_setting(self,):
		w_in_0 = np.zeros((self.r_num,self.u_num))
		ran = (self.beta_i * self.r_num * self.u_num)
		half = int(ran/2)
		w_in_0[:half] = 1
		w_in_0[half:int(ran)] = -1
		np.random.shuffle(w_in_0)
		w_in = self.alpha_i * w_in_0
		self.w_in = w_in#(r_num,u_num)


	def fai(self,t):
		fai = np.heaviside(np.sin(2*np.pi*t),1)
		return fai

	def c_s(self,t):
		c_s = np.array([self.fai(t)]) 
		return c_s

	def u_encoding(self,u):
		u_s = []

		for t in self.time:
			u_s.append([self.fai(t-(1/2)*u[math.floor(t)])])
		return u_s #(60000)
		#print(self.u_s.size)

	def update_r_s(self,r_x):
		r_s = self.r_s
		idx = (r_x >= 1)
		idx2 = (r_x <= 0)
		r_s[idx] = 1
		r_x[idx] = 1

		r_s[idx2] = 0
		r_x[idx2] = 0
		return r_s,r_x

	def run_network(self,):
		for t in self.time:
			t_ = int(t*self.t_div)

			if t%1==0:
				r_t_floor = self.r_s
			if t_ %1000 == 0:
				print(t_)

			#Iの計算
			
			I = np.zeros([self.r_num,1])#(100,1)
			I += self.w_rec @ np.array([self.r_s]).T - 1
			#print(self.u_s[:,t_])
			I += np.array([self.w_in  @ (2*self.u_s[:,t_] - 1)]).T#w_in(100,2),u_s(2,1)

			#Jの計算
			J = self.alpha_s * (np.array([self.r_s]) - self.c_s(t)) * (2*r_t_floor -1)

			#r_xを更新する。
			diff = np.array([1-2*self.r_s])

			dr_x = diff * (1 + np.exp(diff*(I.T+J)/self.temperature)) * t
			r_x = np.array(self.prev_r_x) + dr_x
			
			#r_xが更新されたので、r_sが変化するか調べる。
			r_s,r_x = self.update_r_s(r_x[0])
			

			
			#r_x,r_sを直前と現在だけを記憶しているバージョンを作る。
			self.r_x_collect[t_] = self.prev_r_x
			self.r_s_collect[t_] = self.prev_r_s
			self.J_collect[t_] = J
			self.prev_r_x = self.r_x
			self.prev_r_s = self.r_s

			

			self.r_x = r_x
			self.r_s = r_s
			self.r_decoding(t_)
			"""
			#r_x,r_s記録バージョン
			self.r_x[t_] = r_x
			self.r_s[t_] = r_s

			
			"""

		
	def r_decoding(self,t):
		falling = np.zeros(self.r_num)
		rising = np.zeros(1)

		#get falling_edge
		
		idx = (self.prev_r_s == 1)
		idx2 = (self.r_s[idx] == 0)
		falling[idx] = t/self.t_div

		#get rising_edge
		if self.c_s(t/self.t_div) == 0 and self.c_s(t/self.t_div+1) == 1:
			rising = t/self.t_div
		r_n = np.array([2*(falling - rising) - 1])

		#R
		if self.T0  <=t/self.t_div:
			if t%self.t_div == 0:
				self.R[int(t/self.t_div)-self.T0] = r_n
	




	def w_out_computing(self,):
		
		M = self.R #(250,100)
		M = M
		self.u = self.u[:,self.T0*self.t_div::self.t_div]
		G = np.arctanh(self.u).T #(60000,)
		print(G.shape)
		self.w_out = (np.linalg.inv(M @ M.T + self.lambda0 * self.E)@ M).T @ G#(100,2)にする


	def predict(self,):
		self.y = self.fy(self.w_out.T @ self.R.T)
		cs = self.c_s(self.time)

		plt.plot(list(range(600)),self.r_x_collect[:600,0],lw = 1)
		#plt.plot(list(range(600)),self.r_s_collect[:600,],lw = 0.5,color = 'b')
		plt.title('r_s and r_x')
		plt.show()

		plt.plot(list(range(600)),self.J_collect[:600,0],lw = 1,color = 'y')
		plt.title('J')
		plt.show()

		plt.plot(list(range(600)),cs[0,:600],lw = 0.1,color = 'b')
		plt.title('c_s')
		plt.show()

		plt.plot(list(range(self.T0,self.test_time)),self.y[0])
		plt.plot(list(range(self.T0,self.test_time)),self.y[1])
		plt.title('predict ')
		plt.show()

	def train(self,):

		#input
		self.u_init_generater()

		#w_rec, w_in,c_sを作成する
		self.w_rec_setting()
		self.w_in_setting()

		
		#u(n) encoding u_s(n)
		for i in self.u:
			self.u_s = np.append(self.u_s,self.u_encoding(i))
		self.u_s = self.u_s.reshape(self.u_num,self.t_div*self.test_time)
		#print(self.u_s.shape)

		#network run and decoding
		self.run_network()

		#compute w_out and redge regre
		self.w_out_computing()

		self.predict()


		



model = reservoir_billiard_system()
model.train()

"""
plt.plot(model.time,model.u[1])
plt.plot(model.time,model.u[0])
plt.show()
"""
"""

model.c_s()

c_s = model.c_s
s = model.r_s
x = model.r_x
print(c_s.shape)
plt.scatter([range(60000)],c_s,s = 0.1)
plt.show()


"""










