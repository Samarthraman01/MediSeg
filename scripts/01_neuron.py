rupee = 5
weight = 0.5
bias = 0.1

prediction = rupee*weight + bias
print(f"inout: {rupee} rupees")
print(f"prediction: {prediction}")

correct = 5

loss = (prediction - correct)**2
print(f"loss; {loss}")


learning_rate = 0.01

for step in range(100):
    prediction = rupee*weight + bias
    loss = (prediction - correct)**2
    gradient = 2*(prediction - correct)*rupee
    weight = weight - learning_rate*gradient
    
    if step%10 == 0:
        print(f"Step {step:3d} - loss: {loss:.4f} weight: {weight:.4f} prediction: {prediction:.4f}")
    
print(f"\nFinal weight: {weight:.4f}")
print(f"Final prediction: {rupee * weight + bias:.4f}")
print(f"Correct answer: {correct}")
