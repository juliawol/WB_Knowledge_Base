# WB_Knowledge_Base
This repository contains an approach to creating a knowledge base information retrieval system.
For desciption in Russian, please scroll down.

# Question Answering Model with Gradio Interface

A fine-tuned `SentenceTransformer` model retrieves relevant chunks from a knowledge base based on the input question and uses an optional cross-encoder for re-ranking. The interface is built using [Gradio](https://gradio.app/) to make it easy for users to ask questions and receive relevant answers.

# Structure
- `Data` contains the necessary preprocessed csv-files:
      -  `chunks.csv` contains the knowledge base, cleaned of duplicates, errors, and typos. In some cases, chunks have been redistributed.
      - `train_data.csv` is the training file containing three columns: question, chunk, and hard negative.
      - `train_data1.csv` is the validation dataset.
- `Notebooks` folder contains copies of notebooks from Google Colab, where model fine-tuning was performed using cloud GPU.
      - `WB_Baseline.ipynb` – baseline model for evaluating the potential for metric improvements.
      - `Fine_tuning_with_triplets.ipynb` – code for fine-tuning the model.
      - `WB_Triplets.ipynb` – final model, using the fine-tuned file as a foundation.
      - `Deploy_to_Gradio.ipynb` – code for deploying the model on Gradio.
      - `Recall.ipynb` – code for measuring metrics (also duplicated in the final model for easier verification).

# Features

- **Fine-tuned Model with Triplet Loss**: The model is trained using a `SentenceTransformer` with triplet loss, fine-tuned on custom terminology data.
- **Cross-Encoder Re-ranking**: The system uses a cross-encoder for enhanced answer re-ranking, improving the relevance of returned answers.
- **Gradio Interface**: An easy-to-use web interface built with Gradio, allowing users to ask questions and view the top-5 relevant answers.

# Preprocessing

Preprocessing proved to be a crucial step. Knowledge base and train data contained inconsistencies, repetitions, mistakes and misspellings. A cleaned `chunks.csv` was used to create new train data completed with hard negatives. Given the time constraints, this step unfortunately meant reducing the train data size.

# Fine-tuning

The fine-tuning portion was completed using triplet loss with SentenceTransformer to improve the model's understanding of the similarity between questions and relevant answers (chunks) from the knowledge base

You can refer to the code in `Fine_tuning_with_triplets.ipynb`.
The fine-tuned model is available at https://drive.google.com/file/d/10OcyHBcTVFt4vG8bEJ46i5Ig-OTkSOG_/view?usp=sharing or at HuggingFace JuliaWolken/fine_tuned_model_with_triplets

**Prepare the Knowledge Base**:
   For the purposes of fine-tuning, creating a dataset with hard-negatives was an essential step. This code will only be compatible with train sets that contain a question, true chink and a hard-negative.

## Usage

The main code file (`WB_Triplets.ipynb`) contains the code to retrieve answers. Here’s a breakdown of each component:

1. **Load Fine-Tuned Model**: Loads the `SentenceTransformer` model and precomputes embeddings for all chunks in the knowledge base.
2. **Chunk Retrieval with Cross-Encoder**: Uses cosine similarity for initial retrieval and an optional cross-encoder model for re-ranking.
3. **Gradio Interface**: Sets up a Gradio web application to interact with the model.

### Running the Model

During the following three days, the model is available at https://54bac118d6e06c6b04.gradio.live

## Gradio Interface

The Gradio interface accepts a question in Russian and displays the top-5 relevant chunks from the knowledge base as answers. It handles edge cases where:
- The question is empty or too short.
- The question is not in Russian.
- No relevant chunks are found.

Here’s a screenshot of the interface:

![Interface Screenshot](screenshot.png)

### Example Usage

- **Input**: "Что будет, если я отправлю не тот товар на склад? "
- **Output**:

Chunk 1:
Если вы не заметите подмену и отправите на склад не тот товар, из зарплаты вычтут сумму в размере стоимости вещи.

Chunk 2:
Удержание «Неотправленный возврат в коробке на склад». Если вы отправляли товар на склад, но он потерялся в пути, проверьте историю штрихкода и выберите в заявке склад или СЦ, где товар сканировали в последней раз. - Удержание «Подмена товара». Просмотрите историю штрихкода: там будет статус «Неправильное вложение на переупаковке» и название склада или СЦ, где обнаружили подмену — туда нужно будет отправить заявку. - Удержание «Не принятый товар на полку». Если в приходной коробке не оказалось одного из товаров, нужно найти в истории штрихкода последний склад или СЦ, где его сканировали, и отправить туда заявку.

Chunk 3:
При хранении товара не нужно: - Маркировать товар. Если на коробке будут надписи ручкой или маркером, покупатель может пожаловаться и вернуть товар как бракованный. Тогда из выплаты менеджера удержат сумму за брак. - Игнорировать верхние полки. Так вы лишайте склад дополнительного места для заказов. - Хаотично нумеровать места на складе. Ячейки и полки нужно маркировать в логичном порядке — так сможете быстрее находить и выдавать заказы. - Хранить заказы возле батарей. Товары могут испортиться из-за высокой температуры.  - Не соблюдать принцип «1 покупатель — 1 место». Кладите все покупки одного человека в отдельную ячейку, чтобы не тратить время на поиски.

Chunk 4:
Если вы потеряете товар или коробку в пункте выдачи, отправите вещь на склад без штрихкода или товар не вернётся в сортировочный центр после возврата, программа посчитает это за недостачу. Из зарплаты удержат сумму в размере стоимости товара.

Chunk 5:
Если вы не примете товар из приходной коробки, из зарплаты удержат сумму в размере стоимости вещи. Если товара в коробке не оказалось, за это тоже начислят долг.


### Metrics 

`Recall.ipynb` demonstrated the code that I used to verify the metrics. In this particular case, the focus is on recall@1, recall@#, and recall@5. I would like to highlite that the model improves its predictions by its second attempt at choosing the relevant chunk. The knowledge base itself has a lot similar chunks, thus making the metrics a bit less accurate. Sometimes the true chunk is not  semantically sifferent from the first or second proposed by the model. The following metrics are obtained through using an extremely small dataset with triplets (at about 150 entries). If expanded, the metrcis reach 0.8 for recall@1 (checked using large openly available knowledge bases and systetically created triplets).

Recall@1: 0.45
Recall@2: 0.56
Recall@3: 0.62
Recall@5: 0.68

### Baseline model

 `WB_Baseline.ipynb` contains the baseline model. To evaluate if using transformers and fine-tuning with heavy comtutational load indeed demonstated its benefits, first the data was fed to a simple tf-idf with cosine similarity model. The file includes metrics and examples.


# Версия на русском языке 

# WB_Knowledge_Base
Этот репозиторий содержит подход к созданию системы информационного поиска на основе базы знаний.

# Модель для ответов на вопросы с интерфейсом Gradio

Модель `SentenceTransformer`, дообученная с помощью функции тройных потерь, ищет релевантные фрагменты в базе знаний на основе введенного вопроса и использует дополнительный кросс-энкодер для ранжирования. Интерфейс реализован с помощью [Gradio](https://gradio.app/), что позволяет пользователям задавать вопросы и получать релевантные ответы.


# Структура репозитория
- `Data` – хранилище csv-файлов:
   -  `chunks.csv` содержит базу знаний, очищенную от повторов, ошибок, опечаток. В некоторых случаях чанки перераспределены.
   -  `train_data.csv` – это обучающий файл,содержащий три колонки: вопрос, эталонный ответ, неверный ответ.
   -  `train_data1.csv` – валидационный датасет.
-  `Notebooks` содержит копии тетрадок из Google Colab, на облачном GPU производилось дообучение модели.
   - `WB_Baseline.ipynb` – бейзлайн модель для замера роста метрик.
   - `Fine_tuning_with_triplets.ipynb` – код для дообучения модели.
   -  `WB_Triplets.ipynb` – итоговая модель, использующая в основе файн-тьюн файл.
   -  `Deploy_to_Gradio.ipynb` – код для развертывания модели на сервисе
   -  `Recall.ipynb` – код для измерения метрик (также дублирован в итоговой модели для удобства проверки)

# Основные функции

- **Дообученная модель с функцией тройных потерь**: Модель обучена с использованием `SentenceTransformer` и функции тройных потерь, дообучена на данных с терминологией. 
Код дообучения можно найти в файле `Fine_tuning_with_triplets.ipynb`. Дообученная модель доступна по ссылке https://drive.google.com/file/d/10OcyHBcTVFt4vG8bEJ46i5Ig-OTkSOG_/view?usp=sharing

Также моделью можно воспользоваться 
на HuggingFace: JuliaWolken/fine_tuned_model_with_triplets

- **Ранжирование с кросс-энкодером**: Система использует кросс-энкодер для улучшенного ранжирования аутпута, что повышает релевантность возвращаемых ответов.
- **Интерфейс Gradio**: Простой в использовании веб-интерфейс на Gradio, позволяющий пользователям задавать вопросы и видеть 5 наиболее релевантных ответов.

# Предобработка данных

Предобработка оказалась важным шагом. В базе знаний и обучающих данных были замечены несоответствия, повторы, ошибки и опечатки. Очищенный `chunks.csv` был использован для создания нового обучающего набора данных, дополненного сложными негативными примерами. В условиях ограниченного времени этот шаг привел к сокращению объема обучающих данных.

**Подготовка базы знаний**:
Для целей дообучения важным шагом было создание набора данных с использованием сложных негативных примеров. Этот код будет совместим только с обучающими наборами данных, содержащими вопрос, верный фрагмент и сложный негативный пример.

## Использование

Репозиторий содержит код для загрузки модели, поиска ответов и настройки интерфейса Gradio. Вот краткое описание компонентов:

1. **Загрузка дообученной модели**: Загружает модель `SentenceTransformer` и вычисляет эмбеддинги для всех фрагментов базы знаний.
2. **Поиск фрагментов с кросс-энкодером**: Использует косинусную близость для поиска и кросс-энкодер для дополнительного ранжирования.
3. **Интерфейс Gradio**: Настраивает веб-приложение на Gradio для взаимодействия с моделью.

### Запуск модели

В течение ближайших трех дней модель доступна по адресу https://54bac118d6e06c6b04.gradio.live

## Интерфейс Gradio

Интерфейс Gradio принимает вопрос на русском языке и показывает 5 наиболее релевантных фрагментов из базы знаний в качестве ответов. Приняты во внимание ситуации, когда:
- Вопрос пустой или слишком короткий.
- Вопрос не на русском языке.
- Релевантные фрагменты не найдены.

Пример интерфейса:

![Скриншот интерфейса](screenshot.png)

### Пример использования

- **Ввод**: "Что будет, если я отправлю не тот товар на склад?"
- **Вывод**:

Фрагмент 1:
Если вы не заметите подмену и отправите на склад не тот товар, из зарплаты вычтут сумму в размере стоимости вещи.

Фрагмент 2:
Удержание «Неотправленный возврат в коробке на склад». Если вы отправляли товар на склад, но он потерялся в пути, проверьте историю штрихкода и выберите в заявке склад или СЦ, где товар сканировали в последний раз. - Удержание «Подмена товара». Просмотрите историю штрихкода: там будет статус «Неправильное вложение на переупаковке» и название склада или СЦ, где обнаружили подмену — туда нужно будет отправить заявку. - Удержание «Не принятый товар на полку». Если в приходной коробке не оказалось одного из товаров, нужно найти в истории штрихкода последний склад или СЦ, где его сканировали, и отправить туда заявку.

Фрагмент 3:
При хранении товара не нужно: - Маркировать товар. Если на коробке будут надписи ручкой или маркером, покупатель может пожаловаться и вернуть товар как бракованный. Тогда из выплаты менеджера удержат сумму за брак. - Игнорировать верхние полки. Так вы лишайте склад дополнительного места для заказов. - Хаотично нумеровать места на складе. Ячейки и полки нужно маркировать в логичном порядке — так сможете быстрее находить и выдавать заказы. - Хранить заказы возле батарей. Товары могут испортиться из-за высокой температуры.  - Не соблюдать принцип «1 покупатель — 1 место». Кладите все покупки одного человека в отдельную ячейку, чтобы не тратить время на поиски.

Фрагмент 4:
Если вы потеряете товар или коробку в пункте выдачи, отправите вещь на склад без штрихкода или товар не вернётся в сортировочный центр после возврата, программа посчитает это за недостачу. Из зарплаты удержат сумму в размере стоимости товара.

Фрагмент 5:
Если вы не примете товар из приходной коробки, из зарплаты удержат сумму в размере стоимости вещи. Если товара в коробке не оказалось, за это тоже начислят долг.


### Метрики

`Recall.ipynb` содержит код, использованный для проверки метрик. В данном случае основное внимание уделено `recall@1`, `recall@3` и `recall@5`. Стоит отметить, что модель улучшает предсказания при второй попытке выбора релевантного фрагмента. База знаний содержит множество схожих фрагментов, что снижает точность, но это не вполне отражает качество. Иногда искомый чанк семантически не отличается от первого или второго предложенного моделью. Приведенные метрики получены с использованием небольшого набора данных с триплетами (около 150 строк). При расширении выборки метрики достигают 0.8 для `recall@1` (проверено с использованием крупных открытых баз знаний и синтетически созданных триплетов).


### Базовая модель

Файл `WB_Baseline.ipynb` содержит базовую модель. Чтобы оценить, продемонстрировало ли использование трансформеров и дообучение с высокой вычислительной нагрузкой свои преимущества, данные сначала были поданы в простую модель tf-idf с косинусным сходством. Файл включает метрики и примеры.

 
