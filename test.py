# import sonnet as snt
# import tensorflow as tf 
import numpy as np

class test_class:
	def __init__(self):
		self.user=None
		self.positive_item=None
		self.negative_item=None

def column(matrix, i):
	return [row[i] for row in matrix]

def load_train_dataset():
	# train_data = np.load('train_data.npy')
	# out=train_data.tolist()

	train_dataset=[]
	# with open('train_data.txt', 'w') as f:
	# 	for item in out:
	# 		train_dataset.append(item)
	# 		f.write("%s\n" % item)
	with open('small_data.txt', 'r') as f:
		for item in f:
			k=[int(e) for e in item[:3].split(',')]
			train_dataset.append(k)
			#f.write("%s\n" % item)

	return train_dataset

def load_test_dataset():
	test_data = np.load('test_data.npy')
	out=test_data.tolist()

	test_dataset=[]
	with open('test_data.txt', 'w') as f:
		for item in out:
			temp=test_class()
			temp.user=item 
			f.write("%s\n" % item)
			c=0
			for j in out[item]:
				if c==0:
					temp.positive_item=j 
				else:
					temp.negative_item=j
				f.write("%s\n" % j)
			test_dataset.append(temp)
			
	return test_dataset

class MF():

	def __init__(self, R, K, alpha, beta, iterations):
		'''
		- R (ndarray)   : user-item rating matrix
		- K (int)       : number of latent dimensions
		- alpha (float) : learning rate
		- beta (float)  : regularization parameter
		'''

		self.R = R
		self.num_users, self.num_items = R.shape
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.iterations = iterations

	def train(self):
		# Initialize user and item latent feature matrice
		self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
		self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

		# Initialize the biases
		self.b_u = np.zeros(self.num_users)
		self.b_i = np.zeros(self.num_items)
		self.b = np.mean(self.R[np.where(self.R != 0)])

		# Create a list of training samples
		self.samples = [
			(i, j, self.R[i, j])
			for i in range(self.num_users)
			for j in range(self.num_items)
			if self.R[i, j] > 0
		]

		# Perform stochastic gradient descent for number of iterations
		training_process = []
		for i in range(self.iterations):
			np.random.shuffle(self.samples)
			self.sgd()
			mse = self.mse()
			training_process.append((i, mse))
			# if (i+1) % 10 == 0:
			#     print("Iteration: %d ; error = %.4f" % (i+1, mse))

		return training_process

	def mse(self):
		"""
		A function to compute the total mean square error
		"""
		xs, ys = self.R.nonzero()
		predicted = self.full_matrix()
		error = 0
		for x, y in zip(xs, ys):
			error += pow(self.R[x, y] - predicted[x, y], 2)
		return np.sqrt(error)

	def sgd(self):
		"""
		Perform stochastic graident descent
		"""
		for i, j, r in self.samples:
			# Computer prediction and error
			prediction = self.get_rating(i, j)
			e = (r - prediction)

			# Update biases
			self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
			self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])

			# Update user and item latent feature matrices
			self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
			self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

	def get_rating(self, i, j):
		"""
		Get the predicted rating of user i and item j
		"""
		prediction = self.b + self.b_u[i] + self.b_i[j] + self.P[i, :].dot(self.Q[j, :].T)
		return prediction

	def full_matrix(self):
		"""
		Computer the full matrix using the resultant biases, P and Q
		"""
		return self.b + self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.dot(self.Q.T)

def dot(K, L):
	if len(K) != len(L):
		return 0

   return [i[0] * i[1] for i in zip(K, L)]

def dot_11(K, L):
	
	mat=[[K[j]*L[i] for i in range(len(K))] for j in range(len(K))]
	
	return mat

def dot_d1(K, L):
	
	mat=[0 for j in range(len(K))]

	for i in range(len(K)):
		for j in range(len(K[i])):
			mat[i]+=K[i][j]*L[i]
	
	return mat

train_dataset=load_train_dataset()
#test_dataset=load_test_dataset()

print(train_dataset[:10])
users_list=sorted(list(set(column(train_dataset,0))))
no_users=len(users_list)
print('Total users:	',no_users)
items_list=sorted(list(set(column(train_dataset,1))))
no_items=len(items_list)
print('Total items:	',no_items)

interaction_matrix=[[0 for i in range(no_items)] for j in range(no_users)]
#[e[:2] for e in a[:2]]


for i in range(len(train_dataset)):
	a=users_list.index(train_dataset[i][0])
	b=items_list.index(train_dataset[i][1])
	interaction_matrix[a][b]=1

x=np.array(interaction_matrix)
print('interaction_matrix:\n',interaction_matrix,'\nShape of Rating Matrix:',x.shape)
d=50
m_factors=MF(x,d,0.0002,0.02,1000)
m_factors.train()

U_memory=m_factors.P
I_memory=m_factors.Q

print(U_memory.shape,I_memory.shape)
#print('\nFull rating',type(m_factors.full_matrix))

generated_interaction_matrix=interaction_matrix
for i in range(no_users):
	for j in range(no_items):
		generated_interaction_matrix[i][j]=m_factors.get_rating(i,j)

print(generated_interaction_matrix)

#qui matrix construction
User_preference=[[None for i in range(no_items)] for j in range(no_users)]

for i in range(len(U_memory)):
	mu = U_memory[i]
	for j in range(len(I_memory)):
		ei = I_memory[j]
		qui = []
		for k in range(len(interaction_matrix)):
			if interaction_matrix[k][j] == 1:
				mv = U_memory[k]
				quiv = dot_11(mu, mv) + dot_11(ei, mv)
				qui.append(quiv)
		User_preference[i][j] = qui

Attention_weights=[[None for i in range(no_items)] for j in range(no_users)]
for i in range(len(U_memory)):
	for j in range(len(I_memory)):
		exps=[]
		for k in range(len(User_preference[i][j])):
			exps.append(list(np.exp(User_preference[i][j][k])))
		sum_exps=[[0 for i in range(d)] for j in range(d)]
		for k in range(len(exps)):
			for i in range(d):
				for j in range(d):
					sum_exps[i][j]+=exps[k][i][j]
		pui=[]
		for k in range(len(exps)):
			#puiv=exps[k]/sum_exps
			puiv=[[0 for i in range(d)] for j in range(d)]
			for i in range(d):
				for j in range(d):
					puiv[i][j]=exps[k][i][j]/sum_exps[i][j]
			pui.append(puiv)
		Attention_weights[i][j]=pui 

C=U_memory
Weighted_sums=[[0 for i in range(no_items)] for j in range(no_users)]
for i in range(len(U_memory)):
	for j in range(len(I_memory)):
		for k in range(len(User_preference[i][j])):
			Weighted_sums[i][j]+=dot_d1(puiv,C[k])

U_matrix=[[0 for i in range(d)] for j in range(d)]
W_matrix=[[0 for i in range(d)] for j in range(d)]

rating_score=[[0 for i in range(no_items)] for j in range(no_users)]



for i in range(no_users):
	for j in range(no_items):
		t1=dot_d1(U,dot(U_memory[i],I_memory[j]))
		t2=dot_d1(W,Weighted_sums[i][j])
		t3=b
		rating_score[i][j]= dot_11(vt,relu(t1+t2+t3))
