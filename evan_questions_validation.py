import json

with open('Evan_questions.json', 'r') as file:
    data = json.load(file)

count = len(data)
print(f"Count: {count}")
binary_count = 0
multiclass_count = 0
for object in data:
    if "binary" not in object or "multiclass" not in object:
        print(object)
        raise ValueError("Missing label")
    
    if object["binary"] == 0 and object["multiclass"] >= 3:
        print(object)
        raise ValueError("Mismatched label values")
    
    if object["binary"] == 1 and object["multiclass"] <= 2:
        print(object)
        raise ValueError("Mismatched label values")
    
    if "binary" in object and "multiclass" in object:
        binary_count += 1
        multiclass_count += 1
    
print(f"Binary Count: {binary_count}, Multiclass Count: {multiclass_count}")