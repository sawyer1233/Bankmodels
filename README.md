Projekt dotyczy klasyfikacji klientów banku na podstawie danych z pliku bank-full.csv. Celem analizy jest przewidzenie wartości zmiennej docelowej y, która informuje, czy klient odpowiedział pozytywnie na kampanię bankową.

W projekcie porównano kilka algorytmów uczenia maszynowego:

KNN,
Naiwny klasyfikator Bayesa,
Drzewo decyzyjne,
Random Forest,
XGBoost,
SVM.

Dla każdego modelu wykonywany jest podobny proces: wczytanie danych, przygotowanie zmiennych, podział danych na zbiór treningowy i testowy, trenowanie modelu, wykonanie predykcji oraz ocena jakości klasyfikacji.

Modele są oceniane za pomocą macierzy pomyłek, krzywych ROC oraz wartości AUC. Dzięki temu można porównać, który algorytm najlepiej radzi sobie z przewidywaniem odpowiedzi klienta.
