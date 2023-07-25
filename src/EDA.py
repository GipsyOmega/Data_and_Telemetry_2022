import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

linear_result = pd.read_csv('linearmodel.csv')
lasso_result = pd.read_csv('lassomodel.csv')
poly_result = pd.read_csv('polymodel.csv')
svr_result = pd.read_csv('svrmodel.csv')

sns.set_style('darkgrid')
#plt.figure(figsize=(17, 12))
plt.subplot(221)
plt.title('Polynomial Regression: 77.62%')
plt.plot(range(len(poly_result['True Value'])),
         poly_result['True Value'], color='black', label='True Value')
plt.plot(range(len(poly_result['True Value'])),
         poly_result['Predicted'], color='red', label='Predicted Value')
plt.legend()
plt.subplot(222)
plt.title('Linear Regression: 82.21%')
plt.plot(range(len(linear_result['True Value'])),
         linear_result['True Value'], color='brown', label='True Value')
plt.plot(range(len(linear_result['True Value'])),
         linear_result['Predicted'], color='green', label='Predicted Value')
plt.legend()
plt.subplot(223)
plt.title('Lasso Regression: 92.09%')
plt.plot(range(len(svr_result['True Value'])),
         svr_result['True Value'], color='green', label='True Value')
plt.plot(range(len(svr_result['True Value'])),
         svr_result['Predicted'], color='magenta', label='Predicted Value')
plt.legend()
plt.subplot(224)
plt.title('Support Vector Regression: 96.12%')
plt.plot(range(len(lasso_result['True Value'])),
         lasso_result['True Value'], color='blue', label='True Value')
plt.plot(range(len(lasso_result['True Value'])),
         lasso_result['Predicted'], color='orange', label='Predicted Value')
plt.legend()
plt.tight_layout()
plt.show()

forest_result = pd.read_csv('forestmodel.csv')
plt.title('Random Forest Regression: 99.85%')
plt.plot(range(len(forest_result['True Value'])),
         forest_result['True Value'], color='red', label='True Value')
plt.plot(range(len(forest_result['True Value'])),
         forest_result['Predicted'], label='Predicted Value')
plt.tight_layout()
plt.legend()
plt.show()
