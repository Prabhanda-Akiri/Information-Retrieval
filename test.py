import numpy as np
import sonnet as snt
import tensorflow as tf 


class test_class:
	def __init__(self):
		self.user=None
		self.positive_item=None
		self.negative_item=None

def column(matrix, i):
    return [row[i] for row in matrix]

def load_train_dataset():
	train_data = np.load('train_data.npy')
	out=train_data.tolist()

	train_dataset=[]
	with open('train_data.txt', 'w') as f:
		for item in out:
			train_dataset.append(item)
			f.write("%s\n" % item)

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

print(train_dataset[:10])
no_users=len(list(set(column(train_dataset,0))))
print(no_users)
no_items=len(list(set(column(train_dataset,1))))
print(no_items)

interaction_matrix=[[0 for i in range(no_items)] for j in range(no_users)]
#print(train_dataset[0][0])

for i in range(len(train_dataset)):
	a=train_dataset[i][0]
	b=train_dataset[i][1]
	interaction_matrix[a][b]=1


# input_users = tf.placeholder(tf.int32, [None], 'UserID')
# input_items = tf.placeholder(tf.int32, [None], 'ItemID')

# embedding_initializers={'embeddings': tf.truncated_normal_initializer(stddev=0.01)}

# user_memory = snt.Embed(no_users,50,embedding_initializers,name='MemoryEmbed')
# item_memory = snt.Embed(no_items,50,embedding_initializers,name="ItemMemory")

# cur_user=user_memory(input_users)
# cur_item=item_memory(input_items)

