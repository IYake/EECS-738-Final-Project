import pickle


pickle_off = open("params.pkl","rb")
emp = pickle.load(pickle_off)
print(emp)
