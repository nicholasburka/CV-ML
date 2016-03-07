import numpy as np
import math
import matplotlib.pyplot as plt


## Input data is a list of n-dimensional points
def kmeans(data, k):

	###########HELPER FUNCTIONS############
	#This function sorts the datapoints to the closest means
	def assign_to_nearest(data, means):
		#cluster data structure could be optimized
		clusters = []
		for mean in means:
			clusters.append([])

		#Iterate through all points, assigning them to the nearest mean
		for point in data:
			min_dist_index = 0
			min_dist = 1000000000000

			for index, mean in enumerate(means):
				#Compute the distance using numpy's norm function
				dist = np.linalg.norm(point - mean)
				if dist < min_dist:
					min_dist_index = index
					min_dist = dist
			clusters[min_dist_index].append(point)

		return clusters
	#####################################
	def move_to_center(clusters, means):
		new_means = np.zeros(means.shape)

		#this could be optimized slightly by using an all-encompassing call to np.average
		for index, cluster in enumerate(clusters):
			new_means[index] = np.average(cluster, axis=0)

		return new_means
	######################################


	assert(isinstance(data, np.ndarray))

	#There are k means with dimensions equivalent to the datapoints'
	means = np.zeros(((k,) + data.shape[1:]))

	#Initialize the means by randomly selecting points from the data as each mean
	for index, mean in enumerate(means):
		means[index] = data[np.random.randint(data.shape[0])]



	means_moved = True
	clusters = assign_to_nearest(data, means)

	#While the algorithm is still converging (the means are moving)
	#repeatedly reassign points to the nearest mean and move the means
	#to the center of their cluster
	while means_moved:

		clusters = assign_to_nearest(data, means)
		new_means = move_to_center(clusters, means)
		means_moved = not (np.array_equal(new_means, means))
		means = new_means
		#visualize_kmeans(clusters, means)

	return clusters, means


def dbscan(data, min_pts, epsilon=False):
	assert(isinstance(data, np.ndarray))

	#if there's no epsilon provided, set it to one quarter of the standard deviation
	if not epsilon:
		epsilon = np.std(data)/4.0
		print "eps: ", epsilon

	##############HELPER FUNCTIONS##################
	#Function that finds the near neighbors of a point in a list of points using the epsilon provided
	def find_neighbors(points, pt):
		n = []
		for point in points:
			if np.linalg.norm(point["point"] - pt["point"]) < epsilon and (not np.array_equal(point["point"], pt["point"])):
				n.append(point)
		return n
	
	#Function that assembles a cluster once a dense point is found
	#Recursively recalls itself whenever a new dense point is found among neighbors
	#(This should be optimized)
	def make_cluster(pt, neighbors, c, points):
		recurse = False
		new_neighbors = []

		for point in neighbors:
			ind = point["index"]
			if not points[ind]["visited"]:
				points[ind]["visited"] = True
				sub_neighbors = find_neighbors(points, points[ind])
				c.append(points[ind])

				#New dense point found
				if len(sub_neighbors) >= min_pts:
					recurse = True
					new_neighbors = neighbors + sub_neighbors
		if recurse:
			make_cluster(pt, new_neighbors, c, points)
	###################

	clusters = []
	#this data copy is used to look for near neighbors
	data_copy = np.copy(data)
	visited = np.zeros(data.shape[0])
	NOISE = 0
	DENSE = 1
	CONNECTED = 2

	points = []
	for index, point in enumerate(data):
		p = {}
		p["point"] = point
		p["index"] = index
		p["visited"] = False
		p["label"] = -1
		points.append(p)

	
	for i in range(data.shape[0]):
		if points[i]["visited"]:
			continue

		points[i]["visited"] = True

		neighbors = find_neighbors(points, points[i])
		if len(neighbors) < min_pts:
			points[i]["label"] = NOISE
		else:
			points[i]["label"] = DENSE
			c = []
			c.append(points[i])
			make_cluster(points[i], neighbors, c, points)
			clusters.append(c)
	clusters = map(lambda y: map(lambda x: x["point"], y), clusters)
	return clusters




def print_kmeans(clusters, means):
	print "\nK MEANS CLUSTERS:"
	for i,m in enumerate(means):
		print "CLUSTER ", i
		print "MEAN == "+ str(means[i]) + ":"
		print clusters[i]
		print "________________"

def visualize_kmeans(clusters, means):
	colors = "bgrcmykw"
	color_index = 0
	plt.figure()
	for index, cluster in enumerate(clusters):
		cluster = np.array(cluster)
		x = cluster[:,0]
		y = cluster[:,1]
		plt.scatter(x, y, c=colors[color_index])
		plt.scatter(means[index,0], means[index,1], c=colors[color_index], s=60)
		color_index += 1
	plt.show()
	#color code here taken from John Mee, http://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib

def print_dbscan(clusters):
	print "\nDBSCAN CLUSTERS:"
	for i,cluster in enumerate(clusters):
		print "CLUSTER ", i
		print cluster
		print "________________"
def visualize_dbscan(clusters, data):
	colors = "bgrcmykw"
	color_index = 0
	plt.figure(0)
	for index, cluster in enumerate(clusters):
		cluster = np.array(cluster)
		x = cluster[:,0]
		y = cluster[:,1]
		plt.scatter(x, y, c=colors[color_index%7])
		plt.ylim([0,1])
		plt.xlim([0,1])
		color_index += 1
	plt.figure(1)
	x = data[:,0]
	y = data[:,1]
	plt.scatter(x,y, c="b")
	plt.ylim([0,1])
	plt.xlim([0,1])
	plt.show()


data = np.random.rand(100, 2)
'''clusters, means = kmeans(data, 6)
print_kmeans(clusters, means)
visualize_kmeans(clusters, means)'''
clusters = dbscan(data, 4)
print_dbscan(clusters)
visualize_dbscan(clusters,data)