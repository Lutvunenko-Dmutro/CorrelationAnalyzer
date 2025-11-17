import pandas as pd
import numpy as np
import random

def generate_realistic_data(num_samples=200):
    """
    V4.0: Генерація даних з СИЛЬНИМИ зв'язками для графіків.
    """
    
    # 1. Генеруємо "Дохід" (Базова змінна)
    # Логнормальний розподіл (як у житті)
    income = np.random.lognormal(mean=10.5, sigma=0.5, size=num_samples)
    income = np.round(income, -2) # Округлюємо до сотень
    
    # 2. Генеруємо "Витрати" (Лінійно залежать від Доходу + малий шум)
    # r буде близько 0.95-0.99
    expenses = income * 0.75 + np.random.normal(0, income * 0.05, num_samples)
    expenses = np.maximum(expenses, 1000) # Не менше 1000

    # 3. Генеруємо "Кількість покупок" (Залежить від Витрат)
    # r буде близько 0.9
    purchases = expenses / 2000 + np.random.normal(0, 2, num_samples)
    purchases = np.round(np.clip(purchases, 1, 50))

    # 4. Генеруємо "Вік" (Слабо залежить від Доходу)
    age = 25 + (income / 5000) + np.random.normal(0, 10, num_samples)
    age = np.round(np.clip(age, 18, 70))

    # 5. Генеруємо "Час на сайті" (Залежить від Покупок)
    time_on_site = 10 + purchases * 2 + np.random.normal(0, 10, num_samples)
    time_on_site = np.round(np.clip(time_on_site, 5, 180))

    # 6. Генеруємо "Оцінка" (Залежить від Часу + малий шум)
    satisfaction = 2.5 + (time_on_site / 60) + np.random.normal(0, 0.3, num_samples)
    satisfaction = np.round(np.clip(satisfaction, 1, 5), 1)

    # Збираємо DataFrame
    df = pd.DataFrame({
        'ID': range(1, num_samples + 1),
        'Вік': age,
        'Дохід': income,
        'Витрати': expenses,
        'Кількість_покупок': purchases,
        'Час_на_сайті_хв': time_on_site,
        'Оцінка_задоволеності': satisfaction
    })

    # Додаємо трохи сміття (NaN і Текст)
    categories = ['Постійний', 'Новий', 'VIP']
    df['Категорія'] = [random.choice(categories) for _ in range(num_samples)]
    
    for col in ['Дохід', 'Витрати']:
        for _ in range(5): # 5 випадкових пропусків
            df.loc[random.randint(0, num_samples-1), col] = np.nan

    return df

if __name__ == "__main__":
    file_name = "real_world_data.csv"
    df = generate_realistic_data(200)
    df.to_csv(file_name, index=False)
    print(f"Файл '{file_name}' створено! Кореляції мають бути високими.")
