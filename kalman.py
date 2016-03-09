import numpy as np

'''
TAKEN FROM UDACITY - ARTIFICIAL INTELLIGENCE FOR ROBOTICS, LESSON 2, SEBASTIAN THRUN


# Write a program that will iteratively update and
# predict based on the location measurements 
# and inferred motions shown below. 


#1D KALMAN FILTER

def update(mean1, var1, mean2, var2):
    new_mean = float(var2 * mean1 + var1 * mean2) / (var1 + var2)
    new_var = 1./(1./var1 + 1./var2)
    return [new_mean, new_var]

def predict(mean1, var1, mean2, var2):
    new_mean = mean1 + mean2
    new_var = var1 + var2
    return [new_mean, new_var]

measurements = [5., 6., 7., 9., 10.]
motion = [1., 1., 2., 1., 1.]
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.

#Please print out ONLY the final values of the mean
#and the variance in a list [mu, sig]. 

# Insert code here
for i, m in enumerate(measurements):
    mu, sig = update(mu, sig, m, measurement_sig)
    mu, sig = predict(mu, sig, motion[i], motion_sig)
    
print [mu, sig]

#MULTIPLE DIMENSION KALMAN FILTER
def kalman_filter(x, P):
    for n in range(len(measurements)):
        
        # measurement update
        measure_vec = matrix([[measurements[n]]])
        error = measure_vec - H*x
        S = H*P*H.transpose() + R
        KG = P*H.transpose()*S.inverse()
        x = x + (KG*error)
        P = (I - KG*H)*P
        # prediction
        x = F*x + u
        P = F*P*F.transpose()
    return x,P

'''

def stateShapeTest(mat):
	return mat.shape == np.zeros(4).shape

class KalmanFilter:
	def __init__(self, dim_state, dim_measure, dim_control, 
					state_transition_mat, observation_mat,
					init_state_estimate):

		cov_error = np.identity(dim_state)*np.array([100]*dim_state) #set init covariance very high, because very uncertain
		estimated_measurement_error = np.identity(dim_measure)*np.array([.3]*dim_measure)
		init_state=np.zeros(dim_state)

		self.state = init_state
		self.control = np.identity(dim_control)

		self.state_transition_mat = state_transition_mat
		self.observation_mat = observation_mat

		self.predicted_state = init_state
		self.covariance = cov_error

		self.estimated_measurement_error = estimated_measurement_error

		self.error = []
		self.covariance_to_measurement = []
		self.previous_covariance = self.covariance

	#The Kalman filter relies here on what knowledge it already has
	def predict(self, control_vec=0):
		if not self.control:
			self.predicted_state = np.dot(self.state_transition_mat, self.state)
		else:
			self.predicted_state = self.state_transition_mat*self.state + self.control*control_vec

		self.predicted_covariance = np.dot(np.dot(self.state_transition_mat, self.covariance),np.transpose(self.state_transition_mat))
		#print "Predicted covariance: ", self.predicted_covariance
		print "Predicted state: ", self.predicted_state
		print "Predicted cov: \n", self.predicted_covariance
		return self.predicted_state, self.predicted_covariance

	#The Kalman filter updates its estimation framework using a new measurement
	def correct(self, measurement_vec):
		#this is known as the "innovation"
		#print "obs mat: ", self.observation_mat
		#print "pred state: ", self.predicted_state
		self.error = self.observation_mat.dot(self.predicted_state)
		self.error = measurement_vec - self.error
		print "measurement: ", measurement_vec
		print "error: ", self.error
		self.covariance_to_measurement = np.squeeze(np.dot(np.dot(self.observation_mat,self.predicted_covariance),np.transpose(self.observation_mat))) + self.estimated_measurement_error
		print "cov to meas: ", self.covariance_to_measurement

	def update(self):
		kalman_gain = np.dot(np.dot(self.covariance, np.transpose(self.observation_mat)),np.linalg.inv(self.covariance_to_measurement))
		inter = np.squeeze((np.dot(kalman_gain,np.transpose(self.error))))
		self.state = self.state + inter
		print "State: ", self.state
		self.covariance = np.dot((np.identity(self.predicted_covariance.shape[0]) - np.dot(kalman_gain,self.observation_mat)), self.covariance)
		#print "Covariance: ", self.covariance
		return self.state, self.covariance

dim_state = 4 #x,y,dx,dy
dim_measure = 2 #x,y
dim_control = 0 #no known controller

state_transition_mat = np.identity(4)
state_transition_mat[0,2] = 1 #these basically say, to get the next state from the current, add dx to x
state_transition_mat[1,3] = 1 #and dy to y
print "state trans: ", state_transition_mat

observation_mat = np.zeros((2,4))
observation_mat[0,0] = 1
observation_mat[1,1] = 1

init_state = np.array([0,0])
measured = init_state

k = KalmanFilter(dim_state, dim_measure, dim_control, state_transition_mat, observation_mat, init_state)

for i in range(100):
	print "\niteration ", i
	k.predict()
	k.correct(measured)
	k.update()
	measured = measured + np.array([1,0]) + np.random.uniform(low=-.3, high=.3, size=(1,2))
	#print "measured ", measured
#print k.state, "\n", k.covariance
