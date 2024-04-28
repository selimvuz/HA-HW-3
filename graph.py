import matplotlib.pyplot as plt

# Data
steps = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
training_loss = [11.2363, 2.5410, 3.2581, 2.2041, 1.9229, 1.9759, 1.9956, 1.9309]
validation_loss = [4.361118, 3.232167, 2.224619, 2.218018, 1.911502, 1.925485, 1.918340, 1.93429]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(steps, training_loss, label='Training Loss', marker='o', color='red')
plt.plot(steps, validation_loss, label='Validation Loss', marker='o', color='blue')
plt.title('Training ve Validation Loss')
plt.xlabel('Adım Sayısı')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()