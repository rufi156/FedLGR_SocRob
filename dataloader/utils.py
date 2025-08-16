import pandas as pd
import os
import torchvision.transforms as transforms
from .imageloader import CustomDataset
from torch.utils.data import random_split, DataLoader
import torch
import numpy as np
import sys
from sklearn.utils import shuffle

sys.path.append('..')
from utils import predict_gen, train, predict_gen_distil


def load_universal(path):
	data = pd.read_csv(path + "/all_data.csv")
	df = pd.DataFrame(columns=data.columns)
	for i in range(250, 1000):
		df = pd.concat([df, (data.loc[data['Stamp'] == i].mean()).to_frame().T], ignore_index=True)
	# df=df.append(data.loc[data['Stamp'] == i].mean(), ignore_index=True)
	return df


def load_augmented(df):
	data_new = pd.DataFrame(columns=df.columns)
	for c in range(10):
		for i in range(df.shape[0]):
			data_new = pd.concat([data_new, df.iloc[i].to_frame().T])
	return data_new


def load_images(path):
	import re
	def sort_nicely(l):
		""" Sort the given list in the way that humans expect.
		"""

		convert = lambda text: int(text) if text.isdigit() else text
		alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
		l.sort(key=alphanum_key)
		return l
	
	
	df = load_universal(path)
	path = path + "/images"
	li = sort_nicely(os.listdir(path))
	data_images = pd.DataFrame(columns=['Stamp', 'path'] + list(df.columns[1:3]) + list(df.columns[-8:]))

	for i in range(len(df)):
		try:
			entry = [float(li[i][:3]),
					 f'{path}/' + li[i],
					 int(df[df['Stamp'] == float(li[i].split('_')[0])]['Using circle']),
					 int(df[df['Stamp'] == float(li[i][:3])]['Using arrow']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Vacuum cleaning']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Mopping the floor']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry warm food']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry cold food']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry drinks']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry small objects (plates, toys)']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Carry big objects (tables, chairs)']),
					 float(df[df['Stamp'] == float(li[i][:3])]['Cleaning (Picking up stuff) / Starting conversation'])
					 ]
			new_df = pd.DataFrame([entry], columns=data_images.columns)
			data_images = pd.concat([data_images, new_df], ignore_index=True)
		except Exception as e:
			print(f"Error Loading FileName {li[i].split('_')[0]}: {e}; Continuing to next.")

	return data_images

def ac_split(path):
	data = load_images(path)
	arrow = data[data['Using arrow'] == 1]
	circle = data[data['Using circle'] == 1]
	return arrow, circle


def task_splitter_circle_arrow(path, n_clients, aug, batch_size=16):
	if aug:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation(10),
			# transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
			# transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	else:
		train_transform = transforms.Compose([
			transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
		])
	test_transform = transforms.Compose([
		transforms.Resize((128, 128)),  # Adjust the size according to your model requirements
		transforms.ToTensor(),
		transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	])

	arrow, circle = ac_split(path)
	train_arrow_cl = []
	train_circle_cl = []

	# Label columns
	y_labels = arrow.columns[-8:]
	
	arrow = arrow.sample(frac=1).reset_index(drop=True)
	circle = circle.sample(frac=1).reset_index(drop=True)
	
	test_circle = circle.iloc[int(circle.shape[0] * 0.75):]
	test_circle = DataLoader(CustomDataset(test_circle, transform=test_transform), batch_size=batch_size, drop_last=True)

	test_arrow = arrow.iloc[int(arrow.shape[0] * 0.75):]
	test_arrow = DataLoader(CustomDataset(test_arrow, transform=test_transform), batch_size=batch_size, drop_last=True)

	test = pd.concat([circle.iloc[int(circle.shape[0] * 0.75):], arrow.iloc[int(arrow.shape[0] * 0.75):]], axis=0)
	test = test.sample(frac=1).reset_index(drop=True)
	test = DataLoader(CustomDataset(test, test_transform), batch_size=batch_size, drop_last=True)

	arrow = arrow.iloc[:int(arrow.shape[0] * 0.75)]
	circle = circle.iloc[:int(circle.shape[0] * 0.75)]
	
	if aug:
		arrow = load_augmented(arrow)
		circle = load_augmented(circle)
		arrow = arrow.sample(frac=1).reset_index(drop=True)
		circle = circle.sample(frac=1).reset_index(drop=True)
		
	size_arrow = int(arrow.shape[0] / n_clients)
	size_circle = int(circle.shape[0] / n_clients)
	for i in range(n_clients):
		train_arrow_cl.append(
			DataLoader(CustomDataset(arrow.iloc[i * size_arrow:(i + 1) * size_arrow], train_transform), batch_size=batch_size,
					   shuffle=True, drop_last=True))
		train_circle_cl.append(
			DataLoader(CustomDataset(circle.iloc[i * size_circle:(i + 1) * size_circle], train_transform), batch_size=batch_size,
					   shuffle=True, drop_last=True))

	return [train_circle_cl, train_arrow_cl], [test_circle, test_arrow], test, y_labels
