import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Example dataset
# Features: Size (sqft), Bedrooms
# Target: Price ($1000s)
data = {
    "Size": [2104, 1600, 2400, 1416, 3000, 1985, 1534, 1427, 1380, 1494, 1940, 2000, 1890, 4478, 1268, 2300, 1320, 1236, 2609, 3031, 1767, 1888, 1604, 1962, 3890, 1100, 1458, 2526, 2200, 2637, 1839, 1000, 2040, 3137, 1811, 1437, 1239, 2132, 4215, 2162, 1664, 2238, 2567, 1200, 852, 1852, 1203],
    "Bedrooms": [3, 3, 3, 2, 4, 4, 3, 3, 3, 3, 4, 3, 3, 5, 3, 4, 2, 3, 4, 4, 3, 2, 3, 4, 3, 3, 3, 3, 3, 3, 2, 1, 4, 3, 4, 3, 3, 4, 4, 4, 2, 3, 4, 3, 2, 4, 3],
    "Price": [399900, 329900, 369000, 232000, 539900, 299900, 314900, 198999, 212000, 242500, 239999, 347000, 329999, 699900, 259900, 449900, 299900, 199900, 499998, 599000, 252900, 255000, 242900, 259900, 573900, 249900, 464500, 469000, 475000, 299900, 349900, 169900, 314900, 579900, 285900, 249900, 229900, 345000, 549000, 287000, 368500, 329900, 314000, 299000, 179900, 299900, 239500]
}
df = pd.DataFrame(data)

# Features and target
X = df[["Size", "Bedrooms"]]
y = df["Price"]

# Train linear regression model
model = LinearRegression()
model.fit(X, y)

# Predictions for plotting
size_range = np.linspace(df["Size"].min(), df["Size"].max(), 20)
bedroom_range = np.linspace(df["Bedrooms"].min(), df["Bedrooms"].max(), 20)
size_grid, bedroom_grid = np.meshgrid(size_range, bedroom_range)
predicted_prices = model.predict(np.c_[size_grid.ravel(), bedroom_grid.ravel()])
predicted_prices = predicted_prices.reshape(size_grid.shape)

# --- 3D Plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter actual data
ax.scatter(df["Size"], df["Bedrooms"], df["Price"], c='blue', marker='o', label="Actual Data")

# Regression surface
ax.plot_surface(size_grid, bedroom_grid, predicted_prices, color='red', alpha=0.5)

ax.set_xlabel("Size (sqft)")
ax.set_ylabel("Bedrooms")
ax.set_zlabel("Price ($)")
ax.set_title("3D Plot: House Price Prediction")

plt.legend()
plt.show()
