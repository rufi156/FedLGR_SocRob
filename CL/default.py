from __future__ import print_function
import torch
import torch.nn as nn
import random
from .utils import AverageMeter, Timer, accuracy
import sys
import torch.utils.data as data
import torch.nn.functional as F
from scipy.stats import pearsonr
import math
from utils import get_parameters, pearson_correlation, train, test, predict, predict_gen, RMSELoss
from torch.utils.data import ConcatDataset, DataLoader
import warnings
from collections import OrderedDict
import numpy as np
from utils import print_memory_usage

warnings.filterwarnings("ignore")

sys.path.append('..')
from metrics.computation import RAMU
from dataloader.outloader import CustomOutputDataset


def Average(lst):
	return sum(lst) / len(lst)


class NormalNN(nn.Module):
	'''
	Normal Neural Network with SGD for classification
	'''
	
	def __init__(self, agent_config, net, params=None, fedroot=False):
		'''
		:param agent_config (dict): lr=float,momentum=float,weight_decay=float,
									schedule=[int],  # The last number in the list is the end of epoch
									model_type=str,model_name=str,out_dim={task:dim},model_weights=str
									force_single_head=bool
									print_freq=int
									gpuid=[int]
		'''
		super(NormalNN, self).__init__()
		# self.Net = Net
		self.log = print if agent_config['print_freq'] > 0 else lambda \
				*args: None  # Use a void function to replace the print
		self.config = agent_config
		if agent_config['gpuid'][0] > 0:
			self.gpu = True
		else:
			self.gpu = False
		# If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
		self.multihead = True if len(self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
		self.model = self.create_model(model=net, params=params, fedroot=fedroot)
		self.criterion_fn = nn.MSELoss()
		# self.criterion_fn = nn.L1Loss()
		if self.gpu:
			self.cuda()
		
		self.init_optimizer()
		self.reset_optimizer = False
		self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
		self.criterion_rmse = RMSELoss()
	
	# Set a interger here for the incremental class scenario
	
	def init_optimizer(self):
		optimizer_arg = {'params': self.model.parameters(),
		                 'lr': self.config['lr'],
		                 'weight_decay': self.config['weight_decay']}
		if self.config['optimizer'] in ['SGD', 'RMSprop']:
			optimizer_arg['momentum'] = self.config['momentum']
		elif self.config['optimizer'] in ['Rprop']:
			optimizer_arg.pop('weight_decay')
		elif self.config['optimizer'] == 'amsgrad':
			optimizer_arg['amsgrad'] = True
			self.config['optimizer'] = 'Adam'
		
		self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'], gamma=0.1)
	
	def create_model(self, model, params=None, fedroot=False):
		if self.gpu:
			model.cuda()
		if params is not None:
			if not fedroot:
				params_dict = zip(model.state_dict().keys(), params)
				state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
				model.load_state_dict(state_dict)
			else:
				params_dict = zip(model.conv_module.state_dict().keys(), params)
				state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
				model.conv_module.load_state_dict(state_dict, strict=True)
		return model
	
	def forward(self, x):
		return self.model.forward(x)
	
	def predict(self, inputs):
		self.model.eval()
		out = self.forward(inputs)
		for t in out.keys():
			out[t] = out[t].detach()
		return out
	
	def validation(self, dataloader):
		# This function doesn't distinguish tasks.
		batch_timer = Timer()
		acc = AverageMeter()
		batch_timer.tic()
		
		orig_mode = self.training
		self.eval()
		for i, (inputs, target, task) in enumerate(dataloader):
			
			if self.gpu:
				with torch.no_grad():
					inputs = inputs.cuda()
					target = target.cuda()
			output = self.predict(inputs)
			
			# Summarize the performance of all tasks, or 1 task, depends on dataloader.
			# Calculated by total number of data.
			acc = accumulate_acc(output, target, task, acc)
		
		self.train(orig_mode)
		
		self.log(' * Val Acc {acc.avg:.3f}, Total time {time:.2f}'
		         .format(acc=acc, time=batch_timer.toc()))
		return acc.avg
	
	def criterion(self, preds, targets, **kwargs):
		loss = self.criterion_fn(preds, targets)
		return loss
	
	def update_model(self, inputs, targets):
		self.optimizer.zero_grad()
		out = self.forward(inputs)
		loss = self.criterion(out, targets)
		loss.backward()
		self.optimizer.step()
		return loss.detach(), out
	
	def learn_batch(self, train_loader, val_loader=None):
		peak_ram = 0
		ramu = RAMU()
		peak_ram = max(peak_ram, ramu.compute("TRAINING"))
		
		if self.reset_optimizer:  # Reset optimizer before learning each task
			self.log('Optimizer is reset!')
			self.init_optimizer()
		
		y_labels = [0, 1, 2, 3, 4, 5, 6, 7]
		
		for epoch in range(self.config['schedule'][-1]):
			data_timer = Timer()
			batch_timer = Timer()
			batch_time = AverageMeter()
			data_time = AverageMeter()
			losses = AverageMeter()
			# pcc = AverageMeter()
			pearson = {y_labels[i]: [] for i in range(len(y_labels))}
			
			# Config the model and optimizer
			self.log('Epoch:{0}'.format(epoch))
			self.model.train()
			self.scheduler.step(epoch)
			for param_group in self.optimizer.param_groups:
				self.log('LR:', param_group['lr'])
			
			# Learning with mini-batch
			data_timer.tic()
			batch_timer.tic()
			self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
			for i, (inputs, targets) in enumerate(train_loader):
				data_time.update(data_timer.toc())  # measure data loading time
				
				if self.gpu:
					inputs = inputs.cuda()
					targets = targets.cuda()
				
				loss, output = self.update_model(inputs, targets)
				inputs_size = inputs.detach().cpu().numpy().size
				targets = targets.detach().cpu().numpy()
				output = output.detach().cpu().numpy()
				peak_ram = max(peak_ram, ramu.compute("TRAINING"))
				# calculate pearson correlation
				# pcc.update(pearson_correlation(target, output), input.size(0))
				for i in range(len(pearson.keys())):
					if len(targets[:, i]) > 1 and len(output[:, i]) > 1:
						temp = pearsonr(targets[:, i], output[:, i])[0]
						if math.isnan(temp):
							temp = pearsonr(targets[:, i], np.random.normal(output[:, i], 0.0000001))[0]
						pearson[i].append(temp)
				
				losses.update(loss, inputs_size)
				
				batch_time.update(batch_timer.toc())  # measure elapsed time
				data_timer.toc()
			
			for k in pearson.keys():
				pearson[k] = sum(pearson[k]) / len(pearson[k])
			print(f"Epoch {epoch + 1} / {self.config['schedule'][-1]} = {losses.avg}")
			# print(f"Loss : Epoch {epoch + 1} / {self.config['schedule'][-1]} = {losses.avg}")
			# print(f"PCC: Epoch {epoch + 1} / {self.config['schedule'][-1]} = {Average(list(pearson.values()))}")
			# Evaluate the performance of current task
			peak_ram = max(peak_ram, ramu.compute("TRAINING"))
			if val_loader is not None:
				self.validation(val_loader)
		return peak_ram
	
	def test(self, test_loader):
		# loss, rmse, pearson
		losses = AverageMeter()
		pcc = AverageMeter()
		rmse = AverageMeter()
		self.model.eval()
		for i, (inputs, target) in enumerate(test_loader):
			if self.gpu:
				inputs = inputs.cuda()
				target = target.cuda()
			output = self.model(inputs)
			losses.update(self.criterion(output, target), inputs.size(0))
			inputs = inputs.detach()
			target = target.detach()
			output = output.detach()
			if self.gpu:
				inputs = inputs.cpu()
				target = target.cpu()
				output = output.cpu()
			pcc.update(pearson_correlation(target, output), inputs.size(0))
			rmse.update(self.criterion_rmse(target, output), inputs.size(0))
		return losses.avg, pcc.avg, rmse.avg
	
	def learn_stream(self, data, label):
		assert False, 'No implementation yet'
	
	def add_valid_output_dim(self, dim=0):
		# This function is kind of ad-hoc, but it is the simplest way to support incremental class learning
		self.log('Incremental class: Old valid output dimension:', self.valid_out_dim)
		if self.valid_out_dim == 'ALL':
			self.valid_out_dim = 0  # Initialize it with zero
		self.valid_out_dim += dim
		self.log('Incremental class: New Valid output dimension:', self.valid_out_dim)
		return self.valid_out_dim
	
	def count_parameter(self):
		return sum(p.numel() for p in self.model.parameters())
	
	def save_model(self, filename):
		model_state = self.model.state_dict()
		if isinstance(self.model, torch.nn.DataParallel):
			# Get rid of 'module' before the name of states
			model_state = self.model.module.state_dict()
		for key in model_state.keys():  # Always save it to cpu
			model_state[key] = model_state[key].cpu()
		print('=> Saving model to:', filename)
		torch.save(model_state, filename + '.pth')
		print('=> Save Done')
	
	def cuda(self):
		# torch.cuda.set_device(self.config['gpuid'][0])
		self.model = self.model.cuda()
		self.criterion_fn = self.criterion_fn.cuda()
		# Multi-GPU
		if len(self.config['gpuid']) > 1:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.config['gpuid'], output_device=self.config['gpuid'][0])
		return self
	
	def load_model(self, statedict):
		self.model.train()
		params_dict = zip(self.model.state_dict().keys(), statedict)
		state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
		self.model.load_state_dict(state_dict)
	
	def get(self):
		return get_parameters(self.model)


def accumulate_acc(output, target, task, meter):
	if 'All' in output.keys():  # Single-headed model
		meter.update(accuracy(output['All'], target), len(target))
	else:  # outputs from multi-headed (multi-task) model
		for t, t_out in output.items():
			inds = [i for i in range(len(task)) if task[i] == t]  # The index of inputs that matched specific task
			if len(inds) > 0:
				t_out = t_out[inds]
				t_target = target[inds]
				meter.update(accuracy(t_out, t_target), len(inds))
	
	return meter

class Storage(data.Dataset):
	"""
	A dataset wrapper used as a memory to store the data
	"""
	
	def __init__(self):
		super(Storage, self).__init__()
		self.storage = []
	
	def __len__(self):
		return len(self.storage)
	
	def __getitem__(self, index):
		return self.storage[index]
	
	def append(self, x):
		self.storage.append(x)
	
	def extend(self, x):
		self.storage.extend(x)
	
	def update(self, li):
		self.storage = li


class Memory(Storage):
	def reduce(self, m):
		self.storage = self.storage[:m]


class LatentGenerativeReplay(nn.Module):
	def __init__(self, agent_config, net, params, Gen, path, client_id):
		super(LatentGenerativeReplay, self).__init__()
		# self.Net = Net
		self.generator = Gen
		self.log = print if agent_config['print_freq'] > 0 else lambda *args: None  # Use a void function to replace the print
		self.config = agent_config
		if agent_config['gpuid'][0] > 0:
			
			self.gpu = True
			self.Device = torch.device("cuda")
		else:
			self.gpu = False
			self.Device = torch.device("cpu")
		# If out_dim is a dict, there is a list of tasks. The model will have a head for each task.
		self.multihead = True if len(self.config['out_dim']) > 1 else False  # A convenience flag to indicate multi-head/task
		self.model = self.create_model(model=net, params=params)
		# self.generator=self,create_generator()
		self.criterion_fn = nn.MSELoss()
		# self.criterion_fn = nn.L1Loss()
		self.init_optimizer()
		if self.gpu:
			self.cuda()
		self.reset_optimizer = False
		self.valid_out_dim = 'ALL'  # Default: 'ALL' means all output nodes are active
		self.criterion = nn.MSELoss()
		# self.criterion = nn.L1Loss()
		self.path = path
		self.client_id = client_id
	
	# Set a interger here for the incremental class scenario
	def get_generator_weights(self):
		return self.generator.state_dict()
	
	def update_model(self, inputs, targets):
		# self.model=copy.deepcopy(model_)
		out = self.forward(inputs)
		loss = self.criterion(out, targets)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.detach(), out
	
	def init_optimizer(self):
		optimizer_arg = {'params': self.model.parameters(),
		                 'lr': self.config['lr'],
		                 'weight_decay': self.config['weight_decay']}
		if self.config['optimizer'] in ['SGD', 'RMSprop']:
			optimizer_arg['momentum'] = self.config['momentum']
		elif self.config['optimizer'] in ['Rprop']:
			optimizer_arg.pop('weight_decay')
		elif self.config['optimizer'] == 'amsgrad':
			optimizer_arg['amsgrad'] = True
			self.config['optimizer'] = 'Adam'
		
		self.optimizer = torch.optim.__dict__[self.config['optimizer']](**optimizer_arg)
		self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.config['schedule'], gamma=0.1)
	
	def create_model(self, model, params=None):
		if self.gpu:
			model.cuda()
		if params is not None:
			params_dict = zip(model.conv_module.state_dict().keys(), params)
			state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
			model.conv_module.load_state_dict(state_dict, strict=True)
		return model
	
	def forward(self, x):
		return self.model.forward(x)
	
	def set_generator_weights(self, weights):
		self.generator.load_state_dict(weights)
	
	def predict_gen(self, net, trainloader, DEVICE, batch_size=16):
		new_pairs = []
		net.eval()  # Set the model to evaluation mode
		net.to(DEVICE)
		with torch.no_grad():
			for inputs, labels in trainloader:
				inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
				outputs = net(inputs)  # Forward pass through the model
				for i in range(len(outputs)):
					new_pairs.append((outputs[i], outputs[i]))
		# batch_size = 1
		new_data = CustomOutputDataset(new_pairs)
		new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
		return new_data_loader
	
	def predict_from_gen(self, net, num_samples, DEVICE, batch_size=16):
		new_pairs = []
		net.eval()  # Set the model to evaluation mode
		net.to(DEVICE)
		with torch.no_grad():
			for i in range(num_samples):
				inputs = torch.randn(1, 64).to(DEVICE)
				outputs = net.decode(inputs)
				labels = self.model.fc_module(outputs)
				for i in range(len(outputs)):
					new_pairs.append((outputs[i], labels[i]))
		# batch_size = 16
		new_data = CustomOutputDataset(new_pairs)
		new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
		return new_data_loader
	
	def predict_from_gen_gen(self, net, num_samples, DEVICE, batch_size=16):
		new_pairs = []
		net.eval()  # Set the model to evaluation mode
		net.to(DEVICE)
		with torch.no_grad():
			for i in range(num_samples):
				inputs = torch.randn(1, 64).to(DEVICE)
				outputs = net.decode(inputs)
				labels = self.model.fc_module(outputs)
				for i in range(len(outputs)):
					new_pairs.append((outputs[i], outputs[i]))
		# batch_size = 16
		new_data = CustomOutputDataset(new_pairs)
		new_data_loader = DataLoader(new_data, batch_size=batch_size, shuffle=True, drop_last=True)
		return new_data_loader
	
	def create_dataset(self, new_data):
		try:
			current_task_reconstucted_data = torch.load(f'{self.path}/{self.client_id}_current_task_reconstucted_data.pth')
		except:
			current_task_reconstucted_data = self.predict_from_gen(self.generator, num_samples=len(new_data) * 16, DEVICE=self.Device, batch_size=16)
			torch.save(current_task_reconstucted_data, f'{self.path}/{self.client_id}_current_task_reconstucted_data.pth')
		# shuffle it with train_loader and return
		# both are dataloaders
		dataset1 = new_data.dataset
		dataset2 = current_task_reconstucted_data.dataset
		# for input, label in new_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		# for input, label in current_task_reconstucted_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		
		# Combine the datasets using ConcatDataset
		combined_dataset = ConcatDataset([dataset1, dataset2])
		
		# Create a DataLoader for the combined dataset with shuffling
		combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)
		return combined_dataloader
	
	def create_dataset_gen(self, new_data):
		try:
			current_task_reconstucted_data = torch.load(f'{self.path}/{self.client_id}_current_task_reconstucted_data_generator.pth')
		except:
			current_task_reconstucted_data = self.predict_from_gen_gen(self.generator, num_samples=len(new_data) * 16, DEVICE=self.Device,
			                                                           batch_size=16)
			torch.save(current_task_reconstucted_data, f'{self.path}/{self.client_id}_current_task_reconstucted_data_generator.pth')
		# shuffle it with train_loader and return
		# both are dataloaders
		dataset1 = new_data.dataset
		dataset2 = current_task_reconstucted_data.dataset
		# for input, label in new_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		# for input, label in current_task_reconstucted_data:
		# 	print(input.shape)
		# 	print(label.shape)
		# 	break
		
		# Combine the datasets using ConcatDataset
		combined_dataset = ConcatDataset([dataset1, dataset2])
		
		# Create a DataLoader for the combined dataset with shuffling
		combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)
		return combined_dataloader
	
	def loss_function(self, recon_x, x, mu, logvar, input_dim):
		# MSE reconstruction loss
		# MSE = F.binary_cross_entropy(recon_x, x.view(-1, input_dim), reduction='sum')
		MSE = F.mse_loss(input=x.view(-1, input_dim), target=recon_x, reduction='mean')
		# KL divergence
		KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return MSE + KLD
	
	def train_generator(self, train_loader, task_count=0):
		print(" ............................................................................ Learning LGR Training Generator")
		
		# create dataset from latent features of self.model.root
		self.generator.train()
		self.generator.to(self.Device)
		optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0001)
		gen_train_data = predict_gen(self.model.conv_module, train_loader, self.Device, batch_size=16)
		if task_count != 0:
			gen_train_data = self.create_dataset_gen(gen_train_data)
		tot = self.config['schedule'][-1]
		for epoch in range(self.config['schedule'][-1] // 2):
			for input, target in gen_train_data:
				if self.gpu:
					input = input.cuda()
					target = target.cuda()
				recon_batch, mu, logvar = self.generator.forward(input)
				loss = self.loss_function(recon_batch, input, mu, logvar, input.shape[1])
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			
			# print loss
			print(f'Epoch [{epoch + 1}/{tot}], Loss: {loss.item():.4f}')
	
	def latent_creator(self, train_loader):
		self.model.eval()
		self.model.to(self.Device)
		current_task_latent_data = predict(self.model.conv_module, train_loader, self.Device, batch_size=16)
		print(" ............................................................................ Learning LGR Data Mixed")
		
		final = self.create_dataset(current_task_latent_data)
		return final
	
	def learn_batch(self, task_count, genweights, train_loader, learn_gen, val_loaderr=None):
		self.generator.load_state_dict(genweights)
		self.task_count = task_count
		self.log("Learning LGR")
		peak_ram = 0
		ramu = RAMU()
		peak_ram = max(peak_ram, ramu.compute("TRAINING"))
		print_memory_usage("at start of learn_batch")
		if self.task_count == 0:
			print(" ............................................................................ Learning LGR Task 0")
			# Config the model and optimizer
			self.model.train()
			params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.001}]
			for epoch in range(self.config['schedule'][-1]):
				print_memory_usage(f"task=0 epoch iteration - epoch{epoch}")
				self.log('Epoch:{0}'.format(epoch))
				data_timer = Timer()
				batch_timer = Timer()
				batch_time = AverageMeter()
				data_time = AverageMeter()
				losses = AverageMeter()
				acc = AverageMeter()
				self.scheduler.step(epoch)
				optimizer = torch.optim.Adam(params)
				#print_memory_usage("task=0 epoch iteration after setup")
				# Learning with mini-batch
				data_timer.tic()
				batch_timer.tic()
				self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
				for i, (inputs, targets) in enumerate(train_loader):
					print_memory_usage(f"task=0 epoch iteration - iterate batches - batch {i}")
					data_time.update(data_timer.toc())
					if self.gpu:
						inputs = inputs.cuda()
						targets = targets.cuda()
					#print_memory_usage("task=0 epoch iteration - before forward")
					out = self.forward(inputs)
					#print_memory_usage("task=0 epoch iteration - after forward")
					loss = self.criterion(out, targets)
					#print_memory_usage("task=0 epoch iteration - after loss")
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					#print_memory_usage("task=0 epoch iteration - after step")
					peak_ram = max(peak_ram, ramu.compute("TRAINING"))
					
					inputs = inputs.detach()
					targets = targets.detach()
					acc = 0
					losses.update(loss, inputs.size(0))
					batch_time.update(batch_timer.toc())
					data_timer.toc()
					#print_memory_usage("task=0 epoch iteration - finish batch")
				print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
				print_memory_usage(f"after epoch {epoch}")
		else:
			print(f" ............................................................................ Learning LGR Task {task_count}")
			
			# train(self.model.fc_module, train_loader, self.Device, self.config['schedule'][-1])
			print_memory_usage(f"task={task_count} before mix gen data")
			mixed_task_data = self.latent_creator(train_loader)
			self.model.train()
			
			for epoch in range(self.config['schedule'][-1]):
				print_memory_usage(f"task={task_count} epoch iteration - epoch{epoch}")
				data_timer = Timer()
				batch_timer = Timer()
				batch_time = AverageMeter()
				data_time = AverageMeter()
				losses = AverageMeter()
				acc = AverageMeter()
				
				# Config the model and optimizer
				self.log('Epoch:{0}'.format(epoch))
				self.scheduler.step(epoch)
				optimizer = torch.optim.Adam(self.model.fc_module.parameters(), lr=0.001)
				# Learning with mini-batch
				data_timer.tic()
				batch_timer.tic()
				self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
				for i, (inputs, targets) in enumerate(mixed_task_data):
					print_memory_usage(f"task={task_count} batch iteration - batch{i}")
					data_time.update(data_timer.toc())
					if self.gpu:
						inputs = inputs.cuda()
						targets = targets.cuda()
					# dont use self.update_model
					# dont use self.forward
					out = self.model.fc_module(inputs)
					loss = self.criterion(out, targets)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					peak_ram = max(peak_ram, ramu.compute("TRAINING"))
					inputs = inputs.detach()
					targets = targets.detach()
					acc = 0
					losses.update(loss, inputs.size(0))
					batch_time.update(batch_timer.toc())
					data_timer.toc()
				print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
				print_memory_usage(f"task={task_count} finish epoch {epoch}")
			# freeze all layers of self.model.fc_module
			print(" ............................................................................ Learning LGR End to End Top Frozen")
			for epoch in range(self.config['schedule'][-1]):
				print_memory_usage(f"task={task_count} EtE epoch iteration - epoch {epoch}")
				data_timer = Timer()
				batch_timer = Timer()
				batch_time = AverageMeter()
				data_time = AverageMeter()
				losses = AverageMeter()
				acc = AverageMeter()
				
				# Config the model and optimizer
				self.log('Epoch:{0}'.format(epoch))
				self.model.train()
				# params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.001}]
				params = [{'params': self.model.conv_module.parameters(), 'lr': 0.00001}, {'params': self.model.fc_module.parameters(), 'lr': 0.0}]
				self.scheduler.step(epoch)
				optimizer = torch.optim.Adam(params)
				
				# Learning with mini-batch
				data_timer.tic()
				batch_timer.tic()
				self.log('Itr\t\tTime\t\t  Data\t\t  Loss\t\tAcc')
				for i, (inputs, targets) in enumerate(train_loader):
					print_memory_usage(f"task={task_count} EtE batch iteration - batch {i}")
					data_time.update(data_timer.toc())
					if self.gpu:
						inputs = inputs.cuda()
						targets = targets.cuda()
					out = self.forward(inputs)
					loss = self.criterion(out, targets)
					optimizer.zero_grad()
					loss.backward()
					optimizer.step()
					
					inputs = inputs.detach()
					targets = targets.detach()
					acc = 0
					losses.update(loss, inputs.size(0))
					batch_time.update(batch_timer.toc())
					data_timer.toc()
				print(f"Epoch {epoch + 1}/{self.config['schedule'][-1]}, Loss: {losses.avg}")
				print_memory_usage(f"task={task_count} EtE finish epoch {epoch}")
		
		if learn_gen == True:
			print_memory_usage(f"task={task_count} before train gen")
			self.train_generator(train_loader, self.task_count)
			print_memory_usage(f"task={task_count} after train gen")
		return peak_ram
