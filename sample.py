import numpy as np
import sonnet as snt
import tensorflow as tf 

def column(matrix, i):
    return [row[i] for row in matrix]

data = np.load('test_data.npy')
#data.shape()
out=data.tolist()
# outfile=open("output.txt","w")
# for item in out:
# 	outfile.write(item)
# outfile.close()

print(out)

# dataset=[]
with open('my_file.txt', 'w') as f:
	for item in out:
		#dataset.append(item)
		f.write("%s\n" % item)
		for j in out[item]:
			f.write("%s\n" % j)


#print(np.ndim(data))
#print(data[-10:])
# print(dataset[:10])
# no_users=len(list(set(column(dataset,0))))
# print(no_users)
# no_items=len(list(set(column(dataset,1))))
# print(no_items)

# input_users = tf.placeholder(tf.int32, [None], 'UserID')
# input_items = tf.placeholder(tf.int32, [None], 'ItemID')

# embedding_initializers={'embeddings': tf.truncated_normal_initializer(stddev=0.01)},
# user_memory = snt.Embed(no_users, 50,embedding_initializers,
#                                      name='MemoryEmbed')
# item_memory = snt.Embed(no_items,50,embedding_initializers,
#                                      name="ItemMemory")
# cur_user=user_memory(input_users)
# cur_item=item_memory(input_items)

