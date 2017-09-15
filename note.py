import theano
import theano.tensor as T
import numpy as np
from theano.tensor.nnet.nnet import sigmoid
import math

class Model:

	'''
	A "model" is a recurrent neural network (RNN).
	'''


	def __init__(self,mode,out_layer,idn,hdn,odn,rate,filename="param.txt"):

		'''
		param:
			mode: "1"=train a new network, "2"=load a saved network from "filename".
			out_layer: can be 'softmax', 'linear' or 'sigmoid'
			idn,hdn,odn: neuron number of input/hidden/output layer.
			rate: learning rate.
		'''

		self.inp_dim=idn
		self.hid_dim=hdn
		self.out_dim=odn
		self.learning_rate=rate
		self.bound=20/math.sqrt(self.inp_dim)

		if mode==1:
			self.u=theano.shared((np.random.random((self.inp_dim,self.hid_dim))-0.5)*self.bound)
			self.w=theano.shared((np.random.random((self.hid_dim,self.hid_dim))-0.5)*self.bound)
			self.v=theano.shared((np.random.random((self.hid_dim,self.out_dim))-0.5)*self.bound)	

		if mode==2:
			fp=open(filename,'r')
			lines=fp.readlines()
			val=lines[0].split()
			self.inp_dim=int(val[0])
			self.hid_dim=int(val[1])
			self.out_dim=int(val[2])
			self.u=np.zeros((self.inp_dim,self.hid_dim))
			self.w=np.zeros((self.hid_dim,self.hid_dim))
			self.v=np.zeros((self.hid_dim,self.out_dim))
			curr=1
			for i in xrange(self.inp_dim):
				for j in xrange(self.hid_dim):
					self.u[i][j]=float(lines[curr])
					curr+=1
			for i in xrange(self.hid_dim):
				for j in xrange(self.hid_dim):
					self.w[i][j]=float(lines[curr])
					curr+=1
			for i in xrange(self.hid_dim):
				for j in xrange(self.out_dim):
					self.v[i][j]=float(lines[curr])
					curr+=1
			self.u=theano.shared(self.u)
			self.w=theano.shared(self.w)
			self.v=theano.shared(self.v)

		x=T.matrix()
		hm1=T.matrix()
		h=sigmoid(x.dot(self.u)+hm1.dot(self.w))
		if out_layer=='softmax':
			o=T.nnet.softmax(h.dot(self.v))
		elif out_layer=='linear':
			o=h.dot(self.v)
		elif out_layer=='sigmoid':
			o=sigmoid(h.dot(self.v))
		self.forward_pass=theano.function(inputs=[x,hm1],outputs=[o,h])
		y=T.matrix()
		if out_layer=='softmax':
			loss=T.sum(-y*T.log(o))
		elif out_layer=='linear' or out_layer=='sigmoid':
			loss=T.sum(T.pow(o[0]-y[0],2))
		du=T.grad(loss,self.u)
		dw=T.grad(loss,self.w)
		dv=T.grad(loss,self.v)
		self.update=theano.function(inputs=[x,hm1,y],updates=[(self.u,self.u-self.learning_rate*du),\
															  (self.w,self.w-self.learning_rate*dw),\
															  (self.v,self.v-self.learning_rate*dv)],)

	def generateInitHidden(self):

		'''
		Generates the initial hidden unit of the RNN.
		'''
		return (np.random.random((1,self.hid_dim))-0.5)*self.bound


	def generateInputLayer(self,label):

		'''
		Turns a label to a input vector.
		i.e. model.generateInputLayer(label=0) -> [1,0,0,0,0]
		'''
		res=np.zeros((1,self.inp_dim))
		res[0][label]=1
		return res


	def generateDesired(self,label):
		return self.generateInputLayer(label)


	def saveToFile(self,filename):

		print('Saving to disk:')
		fp=open(filename,'w')
		fp.write(str(self.inp_dim)+' '+str(self.hid_dim)+' '+str(self.out_dim)+'\n')
		ur=self.u.eval()
		wr=self.w.eval()
		vr=self.v.eval()
		for i in xrange(self.inp_dim):
			for j in xrange(self.hid_dim):
				fp.write(str(ur[i][j])+'\n')
		for i in xrange(self.hid_dim):
			for j in xrange(self.hid_dim):
				fp.write(str(wr[i][j])+'\n')
		for i in xrange(self.hid_dim):
			for j in xrange(self.out_dim):
				fp.write(str(vr[i][j])+'\n')
		fp.close()
		print('Saved.')