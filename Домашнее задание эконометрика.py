#!/usr/bin/env python
# coding: utf-8

# # Домашняя работа Тванба Лауры 2 мэо 2 

# # Задача 2.1.1

# In[10]:


import pandas as pd
import statsmodels.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
#Скачиваем датасет#
data=pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/sleep75.csv')
#Присвоили переменные и добавили константу#
x=data.totwrk
y=data.sleep
x=smf.add_constant(x)

#Создали оптимальную прямую#
model = smf.OLS(y, x).fit()

print('Коэффицент totwrk с константой=',round(model.params['totwrk'],2))
print('const=',round(model.params['const'],2))

#График зависимости sleep от totwrk"
sns.regplot(x='totwrk', y='sleep', data=data)

#Решение задачи без константы#
x=data.totwrk
y=data.sleep
model = smf.OLS(y, x).fit()
print('Коэффицент totwrk без константы=',round(model.params['totwrk'],2))

#Отоброжения графика#
plt.show()


# # Задача 2.1.2

# In[12]:


import pandas as pd
import statsmodels.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
#Скачиваем датасет#
data=pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/sleep75.csv')
#Присвоили переменные и добавили константу#
x=data.age
y=data.sleep
x=smf.add_constant(x)

#Создали оптимальную прямую#
model = smf.OLS(y, x).fit()

print('Коэффицент age с константой=',round(model.params['age'],2))
print('const=',round(model.params['const'],2))

#График зависимости sleep от age"
sns.regplot(x='age', y='sleep', data=data)

#Решение задачи без константы#
x=data.age
y=data.sleep
model = smf.OLS(y, x).fit()
print('Коэффицент age без константы=',round(model.params['age'],2))

#Отоброжения графика#
plt.show()


# # Задача 2.1.3

# In[8]:


import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Скачиваем датасет#
url = 'https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/sleep75.csv'
data = pd.read_csv(url)

#Создали параболу#
model = smf.ols('sleep ~ totwrk + I(totwrk**2)', data=data).fit()

print('Коэффицент totwrk=',round(model.params['totwrk'],2))
print('Коэффицент I(totwrk**2)=',round(model.params['I(totwrk ** 2)'] , 2))
print('const=',round(model.params['Intercept'],2))

# график
sns.scatterplot(x='totwrk', y='sleep', data=data)
plt.plot(data['totwrk'], model.predict(data), color='red', label='парабола')
plt.legend()
plt.show()


# # Задача 2.1.4

# In[13]:


import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#Скачиваем датасет#
url = 'https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/sleep75.csv'
data = pd.read_csv(url)

#Создали параболу#
model = smf.ols('sleep ~ age + I(age**2)', data=data).fit()

print('Коэффицент age=',round(model.params['age'],2))
print('Коэффицент I(age**2)=',round(model.params['I(age ** 2)'] , 2))
print('const=',round(model.params['Intercept'],2))

# график
sns.scatterplot(x='age', y='sleep', data=data)
plt.plot(data['age'], model.predict(data), color='red', label='парабола')
plt.legend()
plt.show()


# # Задача 2.1.5

# In[17]:


import pandas as pd
import statsmodels.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
#Скачиваем датасет#
data=pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/sleep75.csv')
#Присвоили переменные и добавили константу#
x=data[['totwrk', 'age']]
y=data.sleep
x=smf.add_constant(x)

#Создали оптимальную прямую#
model = smf.OLS(y, x).fit()

print('Коэффицент totwrk=',round(model.params['totwrk'],2))
print('Коэффицент age=',round(model.params['age'],2))
print('const=',round(model.params['const'],2))



# # Задача 2.2.1

# In[27]:


import pandas as pd
import statsmodels.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
#Скачиваем датасет#
data=pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/Labour.csv')
#Присвоили переменные и добавили константу#
x=data.capital
y=data.output
x=smf.add_constant(x)

#Создали оптимальную прямую#
model = smf.OLS(y, x).fit()

print('Коэффицент capital с константой=',round(model.params['capital'],2))
print('const=',round(model.params['const'],2))

#График зависимости output от capital"
sns.regplot(x='capital', y='output', data=data)

#Решение задачи без константы#
x=data.capital
y=data.output
model = smf.OLS(y, x).fit()
print('Коэффицент capital без константы=',round(model.params['capital'],2))

#Решаем log(output) на log(capital)
data['log_output'] = np.log(data['output'])
data['log_capital'] = np.log(data['capital'])
#Присвоили переменные и добавили константу#
x=data.log_capital
y=data.log_output
x=smf.add_constant(x)



plt.scatter(data['log_capital'], data['log_output'])
model = smf.OLS(data['log_output'], x).fit()
plt.plot(data['log_capital'], model.predict(), c='r')
print('Коэффицент log_capital с константой=',round(model.params['log_capital'],2))
print('const=',round(model.params['const'],2))
#График зависимости log_output от log_capital"
sns.regplot(x='log_capital', y='log_output', data=data)

#Решаем log(output) на log(capital)
data['log_output'] = np.log(data['output'])
data['log_capital'] = np.log(data['capital'])
#Присвоили переменные и добавили константу#
x=data.log_capital
y=data.log_output


plt.scatter(data['log_capital'], data['log_output'])
model = smf.OLS(data['log_output'], x).fit()
plt.plot(data['log_capital'], model.predict(), c='r')
print('Коэффицент log_capital без константы=',round(model.params['log_capital'],2))

#Отоброжения графика#
plt.show()





# # Задача 2.2.2
# 

# In[28]:


import pandas as pd
import statsmodels.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
#Скачиваем датасет#
data=pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/Labour.csv')
#Присвоили переменные и добавили константу#
x=data.labour
y=data.output
x=smf.add_constant(x)

#Создали оптимальную прямую#
model = smf.OLS(y, x).fit()

print('Коэффицент labour с константой=',round(model.params['labour'],2))
print('const=',round(model.params['const'],2))

#График зависимости output от capital"
sns.regplot(x='labour', y='output', data=data)

#Решение задачи без константы#
x=data.labour
y=data.output
model = smf.OLS(y, x).fit()
print('Коэффицент labour без константы=',round(model.params['labour'],2))

#Решаем log(output) на log(capital)
data['log_output'] = np.log(data['output'])
data['log_labour'] = np.log(data['labour'])
#Присвоили переменные и добавили константу#
x=data.log_labour
y=data.log_output
x=smf.add_constant(x)



plt.scatter(data['log_labour'], data['log_output'])
model = smf.OLS(data['log_output'], x).fit()
plt.plot(data['log_labour'], model.predict(), c='r')
print('Коэффицент log_labour с константой=',round(model.params['log_labour'],2))
print('const=',round(model.params['const'],2))
#График зависимости log_output от log_labour"
sns.regplot(x='log_labour', y='log_output', data=data)

#Решаем log(output) на log(labour)
data['log_output'] = np.log(data['output'])
data['log_labour'] = np.log(data['labour'])
#Присвоили переменные и добавили константу#
x=data.log_labour
y=data.log_output


plt.scatter(data['log_labour'], data['log_output'])
model = smf.OLS(data['log_output'], x).fit()
plt.plot(data['log_labour'], model.predict(), c='r')
print('Коэффицент log_labour без константы=',round(model.params['log_labour'],2))

#Отоброжения графика#
plt.show()


# # Задача 2.2.3

# In[31]:


import pandas as pd
import numpy as np
import statsmodels.api as smf
import matplotlib.pyplot as plt

# Загружаем датасет
data = pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/Labour.csv')

# Регрессия log_output на log_capital, log2_capital
data['log_output'] = np.log(data['output'])
data['log_capital'] = np.log(data['capital'])
data['log2_capital'] = np.log(data['capital'])**2
x = data[['log_capital', 'log2_capital']]
x = smf.add_constant(x)
model = smf.OLS(data['log_output'], x).fit()

print('Коэффицент log_capital=',round(model.params['log_capital'],2))
print('Коэффицент log2_capital=',round(model.params['log2_capital'] , 2))
print('const=',round(model.params['const'],2))

# График
plt.scatter(data['log_capital'], data['log_output'])
x_vals = np.linspace(data['log_capital'].min(), data['log_capital'].max(), 100)
y_vals = model.params['const'] + model.params['log_capital'] * x_vals + model.params['log2_capital'] * x_vals**2
plt.plot(x_vals, y_vals, color='red')
plt.xlabel('log(capital)')
plt.ylabel('log(output)')
plt.show()


# # Задача 2.2.4

# In[32]:


import pandas as pd
import numpy as np
import statsmodels.api as smf
import matplotlib.pyplot as plt

# Загружаем датасет
data = pd.read_csv('https://raw.githubusercontent.com/artamonoff/Econometrica/master/python-notebooks/data-csv/Labour.csv')

# Регрессия log_output на log_capital, log2_capital
data['log_output'] = np.log(data['output'])
data['log_labour'] = np.log(data['labour'])
data['log2_labour'] = np.log(data['labour'])**2
x = data[['log_labour', 'log2_labour']]
x = smf.add_constant(x)
model = smf.OLS(data['log_output'], x).fit()

print('Коэффицент log_labour=',round(model.params['log_labour'],2))
print('Коэффицент log2_labour=',round(model.params['log2_labour'] , 2))
print('const=',round(model.params['const'],2))

# График
plt.scatter(data['log_labour'], data['log_output'])
x_vals = np.linspace(data['log_labour'].min(), data['log_labour'].max(), 100)
y_vals = model.params['const'] + model.params['log_labour'] * x_vals + model.params['log2_labour'] * x_vals**2
plt.plot(x_vals, y_vals, color='red')
plt.xlabel('log(labour)')
plt.ylabel('log(output)')
plt.show()


# In[ ]:




