import pickle

result_addr = f"inference_outputs/prediction_labels/prediction.data"

file = open(result_addr, 'rb') 
data = pickle.load(file) 

print(data)