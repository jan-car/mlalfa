# %% [markdown]
# # Titanic Dataset
#
# <img src="titanic.png">
#
# Fragestellung: Wie können wir effizient vorhersagen, wer überlebt hat und wer nicht? Stimmt die Behauptung, dass
# Frauen und Kinder zuerst gerettet wurden? Welche Eigenschaftn entscheiden am ehesten über das Überleben?
# ------------
# *__Schwerpunkte__*:
#
# - __Sabine__: Untersuchung von GridSearch Parametern, Diskussion Feature Engineering, Fachrecherche für Data Insights
#               für Kategorisierungen (Passagierliste, Titel-Bezeichnungen, etc.), Ausarbeitung von Texten
#
# - __Saskia__: Erstellen von Modellen und Scalern, Parameter-Tests, Diskussion Feature Engineering, statistische
#               Überlegungen zum Cross Validation Score
#
# - __Jan__: Explorative Datenanalyse, Pipeline-Erstellung (Imputer/Transformer/Scaler-Klassen, inklusive
#            Panda-Kompatibilität), Diskussion Feature Engineering, Code Style/Formatierung, Diskussion Modellauswahl
#

# %% [markdown]
# # 1) Imports:
# Zuerst importieren wir notwendige Libraries und viele Elemente aus sklearn:


# %%
# Libraries:
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# Strategic imports:
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
# Machine Learning Models:
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.naive_bayes import CategoricalNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# Setups:
SEED = 42
np.random.seed(SEED)
sns.set()


# %% [markdown]
# # 2) Laden des Datensets:
# Das Datenset wird geladen und erste Eindrücke gewonnen!

# %%
df_train = pd.read_csv('train.csv', index_col='PassengerId')
df_test = pd.read_csv('test.csv', index_col='PassengerId')

# %% [markdown]
# Zunächste betrachten wir die ersten Zeilen des Datensets:
# %%
df_train.head()

# %% [markdown]
# Außerdem lassen wir uns einige Statistiken anzeigen (nur von numerischen Features):
# %%
df_train.describe()

# %% [markdown]
# Das Datenset hat folgende Spalten:
# %%
df_train.info()

# %% [markdown]
# Die Spalten haben folgende Bedeutung:
# * `PassengerId`: Eindeutige Identifikationsnummer des Passagiers, wurde schon als Index der Datensets verwendet und
#   taucht deshalb hier nicht mehr auf.
#
# * `Survived`: Wer hat überlebt? Dies ist unsere Zielspalte (Label).
#
# * `Pclass`: Ticket Klasse (1, 2 oder 3) -> ordinale Skala. Dieses Feature sagt noch nichts darüber aus, wo die Zimmer
#   auf dem Schiff waren (weiter oben an Deck, oder tiefer im Schiff? Dies könnte anhand der `Cabin` erklärt werden.
#
# * `Name`: Name des Passagiers. Dieser enthält auch Titel wie "Mr" oder "Mrs". Bei Frauen kann so vielleicht zwischen
#   verheiratet ("Mrs") und unverheiratet ("Ms") unterschieden werden. Eventuell hat dies einen Einfluss auf die
#   Überlebenswahrscheinlichkeit.
#
# * `Sex`: Entweder "male" oder "female". Sollte vor Benutzung als `0`/`1` kodiert werden.
#
# * `Age`: Alter des Passagiers -> Rationale Skala, aber eventuell ist eine Einteilung in Kategorien sinnvoll?
#
# * `SibSp`: Anzahl der Geschwister (Siblings) und Ehepartner (Spouses).
#
# * `Parch`: Anzahl der Eltern (Parents) und Kinder (Children) des Passagiers.
#
# * `Ticket`: String oder Zahlenfolge, die die Ticketnummer des Passagiers angibt. Eine Ticket Nummer kann sich bei
#   verschiedenen Personen finden, die sich das Ticket also teilen.
#
# * `Fare`: Der Ticketpreis, welcher wahrscheinlich mit Deck (siehe `Cabin`) und Klasse (siehe `Pclass`) korreliert.
#   Scheinbar bezieht sich der Preis auf das Ticket. Die Erstellung eines Features "Preis/Person" scheint daher
#   sinnvoll.
#
# * `Cabin`: Kabinennummer (nur für sehr wenige Passagiere vorhanden). Der Buchstabe steht für das Deck, was eventuell
#   ein wichtiges Indiz für die Evakuierbarkeit des Passagiers zulässt.
#
# * `Embarked`: Hafen, an dem der Passagier an Bord gegangen ist (drei Möglichkeiten: Southampton, Cherbourg,
#   Queenstown). Eventuell korreliert dieser mit der Klasse (Reichtum der Bewohner an den Häfen?). Eventuell wird dieses
#   Feature aber auch weggelassen, da es keinen großen Einfluss auf die Überlebenschancen haben sollte.


# %% [markdown]
# # 3) Explorative Datenanalyse
# Zunächst schauen wir uns die einzelnen Features genauer an. Dazu legen wir erst eine Arbeitskopie des
# Trainingsdatensatzes an und schauen uns außerdem einen Pairplot, sowie die Korrelationsmatrix an:

# %%
df = df_train.copy()
sns.pairplot(df, hue='Survived', kind='scatter', diag_kind='kde', diag_kws={'bw_adjust': 0.5})
plt.show()
sns.heatmap(df.corr(), cmap='seismic_r', annot=True, center=0)
plt.show()

# %% [markdown]
# Auffällig ist schon jetzt, dass die Überlebensrate am stärksten mit der `Pclass` und `Fare` korreliert, welche
# ebenfalls beide korrelieren. Es ist zu beachten, dass kategorische Features (beispielsweise `Sex`) aktuell noch nicht
# in der Korrelationsmatrix auftauchen (dazu müssten sie erst in eine Zahlenskala transformiert werden):

# %%
df.replace({'male': 0, 'female': 1}, inplace=True)
sns.heatmap(df.corr(), cmap='seismic_r', annot=True, center=0)
plt.show()

# %% [markdown]
# Offensichtlich ist die Korrelation mit dem Geschlecht am stärksten!


# %% [markdown]
# ## 3.1) Fehlende Werte
# Ein wichtiger erster Schritt ist festzustellen, in welchem der Features Werte fehlen (`NaN`). Diese müssen dann
# eventuell durch Imputation-Strategien durch sinnvolle Werte ersetzt werden.

# %%
def check_missing_values(df, title=None):
    if title is not None:
        print(title)
    missing_sth = False
    for name in df.columns:
        nan_count = df[name].isnull().values.sum()
        if nan_count > 0:
            missing_sth = True
            print(f'Column "{name}" is missing {nan_count} of {df.shape[0]} values')
    if not missing_sth:
        print('No column has missing data!')
    print()


check_missing_values(df_train, title='Training Set:')
check_missing_values(df_test, title='Test Set:')

# %% [markdown]
# Wir sehen, dass `Age` recht viele fehlende Werte hat, `Embarked` nur zwei Stück und das `Cabin` knapp 3/4 aller Werte
# fehlen!


# %% [markdown]
# ### 3.1.1) Imputation des "Age" Features
# Für die Imputation kann man naiv den Median/Mittelwert aller vorhandenen Werte einsetzen. Wir können jedoch bessere
# Ergebnisse erzielen, wenn wir uns anschauen, welche Werte am besten mit `Age` korrelieren und entsprechend auffüllen:

# %%
df.corr()['Age'].sort_values(ascending=False, key=abs)

# %% [markdown]
# In diesem Fall wäre dies das `Pclass` Feature. Wir könnten also prinzipiell Klassen-Mittelwerte oder Mediane für das
# Alter berechnen und diese für die Imputation nutzen. Wir gehen jedoch einen Schritt weiter und schauen uns ein
# neues Feature an, welches wir aus dem Namen generieren können (mehr dazu in Abschnitt 3.2.2):

# %%
df['Title'] = df['Name'].str.extract(pat='([A-Z][a-z]+\.)')
df['Title'].value_counts()

# %% [markdown]
# Viele Titel kommen nur selten oder ein einziges Mal vor. Wir fassen diese zusammen:

# %%
df['Title'][~df['Title'].isin(['Mr.', 'Miss.', 'Mrs.', 'Master.'])] = 'Misc.'
df['Title'].value_counts()

# %%
df.groupby(['Title'])['Age'].describe()

# %% [markdown]
# Wir sehen, dass die `Title`-Mediane deutliche Unterschiede zeigen, was unseren Ansatz bestätigt, den Titel zur
# Imputation zu nutzen. Im Folgenden füllen wir die fehlenden Werte mit diesen Medianen auf:

# %%
df['Age'].fillna(df.groupby('Title')['Age'].transform('median'), inplace=True)

# %% [markdown]
# Um zu zeigen, dass unser Ansatz besser als `Pclass`-Mediane oder der Gesamt-Median ist, schauen wir uns diese hier an:

# %%
df.groupby(['Pclass'])['Age'].describe()

# %%
df['Age'].describe()

# %% [markdown]
# Für "Master" (kleine Jungen) wären diese Mediane beispielsweise deutlich schlechter gewesen als die von uns gewählten.

# %%
check_missing_values(df)

# %% [markdown]
# Die Features `Cabin` und `Embarked` haben noch fehlende Werte, wir werden jedoch beide vernachlässigen, was ein
# Imputen überflüssig macht. In `Cabin` fehlen zu viele Werte (auch wenn die Deck-Nummer wahrscheinlich wertvolle
# Informationen enthält) und `Embarked` sollte keinen großen Einfluss auf das Überleben haben.
#
# Vorsichtshalber schauen wir auch einmal in die Kaggle-Test-Daten und stellen fest, das hier, zusätzlich zu fehldenden
# Werten in `Age`, `Cabin` und `Embarked`, auch ein Wert in `Fare` fehlt.

# %%
check_missing_values(df_test)
df_test[df_test['Fare'].isnull()]

# %% [markdown]
# Generell macht es Sinn sich für jedes der Features eine Imputer-Strategie zu überlegen, da man den Test-Datensatz
# idealerweise erst nach der Modell-Wahl bekommt und immer Daten fehlen können.


# %% [markdown]
# ### 3.1.2) Imputer Funktion
# Wir schreiben eine Imputer Funktion um später einfacher fehlende Werte zu ersetzen (Spalten, die nicht weiter benutzt
# werden, werden gedroppt, hier: `Cabin` und `Embarked`):

# %%
class TitanicImputer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame), 'This imputer is designed to work on pandas DataFrames!'
        titles = self._get_titles(X)
        self.median_age_per_title = pd.concat((titles, X['Age']), axis=1).groupby('Title')['Age'].median()
        self.median_fare_per_class = X.groupby('Pclass')['Fare'].median()
        return self

    def transform(self, X, copy=True):
        assert isinstance(X, pd.DataFrame), 'This imputer is designed to work on pandas DataFrames!'
        if copy:
            X = X.copy()
        titles = self._get_titles(X)
        X.loc[X['Age'].isnull(), 'Age'] = titles.map(self.median_age_per_title)
        X.loc[X['Fare'].isnull(),'Fare'] = X['Pclass'].map(self.median_fare_per_class)
        X.drop(['Cabin', 'Embarked'], axis=1, inplace=True)  # Unused columns with nans are dropped instead of filling!
        return X

    @staticmethod
    def _get_titles(X):
        # Extract regex; expand=False: if just one capture group (round brackets), return Series, not DataFrame!
        return X['Name'].str.extract(pat='([A-Z][a-z]+\.)', expand=False).rename('Title')


imputer = TitanicImputer()
df_train_imputed = imputer.fit_transform(df_train)
df_test_imputed = imputer.transform(df_test)
check_missing_values(df_train, title='Training Set:')
check_missing_values(df_test, title='Test Set:')
check_missing_values(df_train_imputed, title='Training Set (Imputed):')
check_missing_values(df_test_imputed, title='Test Set (Imputed):')


# %% [markdown]
# ## 3.2) Sicht der Daten, Generierung neuer und Vernachlässigung unnötiger Features
# Wir wollen nun einmal die einzelnen Features durchgehen und bewerten:


# %% [markdown]
# ### 3.2.1) Ticket Klasse `Pclass`

# %%
sns.countplot(x='Pclass', data=df, hue='Survived')
plt.show()

# %% [markdown]
# Die Ticket Klasse ist ein gutes Indiz für das Überleben der Passagiere. In der dritten Klasse sinken die Chancen das
# Unglück zu überleben drastisch. Dieses Feature hat eine ordinale Skala und kann von uns so weiterverwendet werden.
# Eventuell müssen wir noch skalieren (`StandardScaler` oder `MinMaxScaler` bieten sich an).


# %% [markdown]
# ### 3.2.2) Passagier `Name`

# %%
df['Name'].head(10)

# %% [markdown]
# `Name` enthält den Namen der Passagiere und hat dementsprechend eine Nominalskala. Wir haben stichprobenartig
# untersucht, ob die Cross channel Passagiere (siehe: https://en.wikipedia.org/wiki/Passengers_of_the_Titanic) im
# Datensatz vorkommen (wir haben außergewöhnliche/auffällige Namen genutzt):

# %%
cross_channel_samples = ['DeGrasse', 'Dyer-Edwardes', 'Lenox-Conyngham', 'Osborne', 'Remesch']
if df['Name'].str.contains('|'.join(cross_channel_samples)).any():
    print('(Some) cross channel passengers are included!')
else:
    print('No cross channel passengers were found!')

# %% [markdown]
# Obwohl wir den Namen selbst nicht benutzen können, ist es uns möglich ein interessantes Feature aus dem Datensatz zu
# extrahieren: den Titel der Person! Dieser wird immer groß geschrieben und endet in einem Punkt und wir können ihn mit
# einer Regular Expression erfassen (dieses Feature wurde schon in Abschnitt 3.1.1 zur Imputation benutzt und wird hier
# weiter erklärt):

# %%
df['Title'] = df['Name'].str.extract(pat='([A-Z][a-z]+\.)')
df['Title'].value_counts()

# %% [markdown]
# Sehr viele der selteneren Titel tauchen nur wenige Male auf und werden von uns zu `Misc.` (Miscellaneous =
# Verschiedenes) zusammengefasst:

# %%
df['Title'][~df['Title'].isin(['Mr.', 'Miss.', 'Mrs.', 'Master.'])] = 'Misc.'
df[df['Title'].isin(['Misc.'])].head()

# %%
df['Title'].value_counts()

# %% [markdown]
# Schauen wir uns die `Misc.`-Titel ein wenig genauer im Hinblick auf die Überlebensrate an:
df[df['Title'] == 'Misc.'].sort_values(by='Survived')

# %% [markdown]
# Interessant ist, dass (wie bereits in 3.1.1 gezeigt), das Alter der Passagiere mit seltenen Titeln sehr hoch ist, was
# sich mit dem hohen Rang (Militär) oder geistlichen Würden (z.B. "Rev." für "Reverend"), sowie Adelsstand (z.B.
# "Countess") begründen lässt.

# %%
sns.countplot(x='Sex', data=df[df['Title'] == 'Misc.'], hue='Survived')

# %% [markdown]
# Außerdem kann man sehen, dass alle Frauen dieser Kategorie überlebt haben, von den
# Männern aber überdurchschnittlich viele nicht. Wir schätzen dieses Feature von daher als recht wichtig ein.
#
# Zu guter Letzt wandeln wir die immer noch nominale Skala des Titels in 5 binäre Features um, die von unseren
# Algorithmen verwendet werden können:

# %%
df = pd.get_dummies(df, columns=['Title'])
df.info()


# %% [markdown]
# ### 3.2.3) Geschlecht `Sex`

# %%
sns.countplot(x='Sex', data=df, hue='Survived')
plt.show()

# %% [markdown]
# Das Geschlecht hat einen sehr starken Einfluss auf die Überlebenschancen! Aktuell hat dieses Feature eine Nominalskala
# ("male"/"female") und wird von uns in ein binäres Feature umgewandelt:

# %%
df.replace({'male': 0, 'female': 1}, inplace=True)
df['Sex'].value_counts()


# %% [markdown]
# ### 3.2.4) Alter `Age`
# # Dieses Feature zeigt das Alter der Passagiere an.

# %%
sns.histplot(x='Age', data=df, hue='Survived')

# %% [markdown]
# Aus dem Histogramm lässt sich erkennen, dass junge Kinder viel höhere Überlebenschancen hatten als Erwachsene. Sehr
# auffällig ist der hohe Anteil von Ertrinkenden bei den ca. 30-jährigen. Entsprechend dieser Statistik haben wir uns
# für eine Kategorisierung der Daten entschieden. Ein paar Entscheidungskriterien:
#
# * 5 war das Einschulungsalter zur damaligen Zeit.
#
# * Der älteste "Master" im Datenset ist 12 Jahre alt.
#
# * Volljährigkeit mit 18.
#
# * Peaks im Graph bei ca. 20 und ca. 30 rechtfertigen Abschnitte von 18-25, sowie von 35 bis 45.
#
# * Sehr alte Menschen (über 60) scheinen keine hohen Überlebenschancen zu haben.

# %%
df['AgeCat'] = pd.cut(df['Age'], bins=[0, 5, 12, 18, 25, 35, 60, np.inf], labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)
sns.countplot(x='AgeCat', data=df, hue='Survived')


# %% [markdown]
# ### 3.2.5) `SibSp` und `Parch`
# Diese Features beschreiben die Anzahl der Geschwister (Siblings) und Ehepartner (Spouses), sowie die Anzahl der Eltern
# (Parents) und Kinder (Children):

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
sns.countplot(x='SibSp', data=df, hue='Survived', ax=axes[0])
sns.countplot(x='Parch', data=df, hue='Survived', ax=axes[1])

# %% [markdown]
# Beide Features zeigen einen sehr ähnlichen Zusammenhang zur Überlebensrate und werden von uns von daher zur
# Familiengröße ´FamilySize´ zusammengefasst:

# %%
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1  # +1: die Person selbst wird mit eingerechnet!
df['FamilySize'].value_counts()

# %% [markdown]
# Eine interessante Entdeckung war ein Geschwisterpaar (`SibSp`=1), welches ohne Eltern (`Parch`=0) auf Reisen waren.
# Beide (12 und 14 Jahre) haben überlebt!

# %%
df[df['Ticket']=='2651']


# %% [markdown]
# ### 3.2.6) Ticketnummer `Ticket`
# Die Ticketnummer kann von uns nicht direkt verwendet werden, da es scheinbar kein eindeutiges System für die Zahlen
# und Buchstaben gibt. Man kann weder die Kabine, noch das Deck ableiten und es scheint, dass unterschiedliche
# Ausgabestellen andere Konventionen verwenden. Was wir jedoch tun können, ist die Ticketnummer zu verwenden um
# Passagiere zusammenzufassen, die zusammen gereist sind. Wir führen deshalb das neue Feature `GroupSize` ein welches
# nicht unähnlich zur `FamilySize` ist (allerdings können auch Freunde zusammen reisen und Familien können mehrere
# Tickets nutzen, es besteht also ein Mehrwert dieses Features):

# %%
df['GroupSize'] = df['Ticket'].map(df['Ticket'].value_counts())
df['GroupSize'].value_counts()


# %% [markdown]
# Im Folgenden zeigen wir die ersten drei Reisegruppen mit dem Maximum von 7 Personen in `GroupSize`:
# * `Ticket=CA. 2434`: Familie Sage mit 11 Personen, die also mehrere Tickets besaßen. Alle 7 von diesem Ticket starben.
#
# * `Ticket=347082`: Familie Andersson mit 7 Personen, alle mit diesem Ticket. Alle starben.
#
# * `Ticket=1601`: Eine asiatische Reisegruppe, die nicht verwandt war, von denen 5 überlebten.
#
# Für die erste Gruppe gibt es leider keine gute Möglichkeit die anderen 4 Familienmitglieder (auf mindestens einem
# weiteren Ticket) ausfindig zu machen um zu überprüfen ob diese überlebt haben.

# %%
columns = ['Name', 'Ticket', 'GroupSize', 'FamilySize', 'Survived']
df[columns].sort_values(by=['GroupSize', 'Ticket'], ascending=False).head(21)


# %% [markdown]
# ### 3.2.7) Ticketpreis `Fare`

# %%
df.groupby(['Pclass'])['Fare'].describe()

# %% [markdown]
# Der Ticketpreis variiert sehr stark bis hin zu 512\$ in der 1. Klasse. Auffällig ist auch, dass scheinbar Leute umsonst
# mitgefahren sind (0\$ Minimum in allen 3 Klassen). Ein weiteres Problem ist, dass die Preise pro Ticket und nicht pro
# Person angegeben sind. Dies korrigieren wir im Folgenden:

# %%
df['PersonPerTicket'] = df['Ticket'].map(df['Ticket'].value_counts())
df['FarePerPerson'] = df['Fare'] / df['PersonPerTicket']

# %% [markdown]
# Um weitere Aussagen machen zu können schauen wir auf ein Histogramm:

# %%
sns.histplot(x='FarePerPerson', data=df, hue='Survived')
plt.show()


# %% [markdown]
# Die extrem hohen Preise scheinen sehr starke Ausreißer zu sein, was das Lesen des Plots erschwert. Wir beschränken
# deshalb die Plotting-Range in `x`. Zusätzlich schneiden beschränken wir auch `y` um höhere Preise besser untersuchen
# zu können:

# %%
sns.histplot(x='FarePerPerson', data=df, hue='Survived')
plt.xlim(0, 150)
plt.ylim(0, 50)
plt.show()

# %% [markdown]
# Auffällig viele Passagiere in den niedrigen Preisklassen haben die Reise nicht überlebt. Da der Preis mit der Klasse
# korrelieren sollte, ist dies jedoch nicht verwunderlich. Jedoch scheinen Leute mit sehr hohen Ticketpreisen sehr gute
# Chancen zu haben. Von den Leuten, die 0$ gezahlt haben verunglückten die meisten! Aufgrund dieser Überlegungen legen
# wir auch für `Fare` Kategorien fest, die wir in einem neuen Feature `FareCat` speichern.

# %%
df['FareCat'] = pd.cut(df['FarePerPerson'], bins=[-1, 1, 10, 20, 30, 50, np.inf], labels=[1, 2, 3, 4, 5, 6]).astype(int)
sns.countplot(x='FareCat', data=df, hue='Survived')
plt.show()
df.groupby(['Survived'])['FareCat'].value_counts()

# %% [markdown]
# Unsere Einteilung zeigt sogar, das nur eine Person mit einem Preis von 0\$ überlebt hat!

# %%
df[(df['FareCat'] == 1) & (df['Survived'] == 1)]

# %% [markdown]
# __ANMERKUNG:__
# Wenn nicht alle Passagiere eines Tickets im Trainingsset sind (z.B. teilweise im Testset oder gar nicht vorhanden),
# dann kann von der ermittelten Zahl der Personen pro Ticket nicht exakt auf den pro-Kopf-Preis geschlossen werden.
# Dies wird von uns hier jedoch vernachlässigt.


# %%
#
# TODO: ### 3.2.8) Kabinennummer `Cabin`
# `Cabin`: Kabinennummer (nur für sehr wenige Passagiere vorhanden). Der Buchstabe steht für das Deck, was eventuell
# ein wichtiges Indiz für die Evakuierbarkeit des Passagiers zulässt.
#
# TODO: ### 3.2.9) Starthafen `Embarked`
# `Embarked`: Hafen, an dem der Passagier an Bord gegangen ist (drei Möglichkeiten: Southampton, Cherbourg,
# Queenstown). Eventuell korreliert dieser mit der Klasse (Reichtum der Bewohner an den Häfen?). Eventuell wird dieses
# Feature aber auch weggelassen, da es keinen großen Einfluss auf die Überlebenschancen haben sollte.


# %% [markdown]
# ### 3.2.10) FeatureTransformer Klasse
# Im Folgenden fassen wir unsere Überlegungen über die ursprünglichen Features des Datensets in einer Transformer-Klasse
# zusammen, welche uns vorhandene Features bei Bedarf umformt, neue Features erstellt und zu guter Letzt Features
# droppt, die wir nicht für die Modelle benötigen:

# %%
class TitanicFeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        # TODO: We could use some flags here to customize the transform behaviour!
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, copy=True):
        assert isinstance(X, pd.DataFrame), 'This transformer is designed to work on pandas DataFrames!'
        if copy:
            X = X.copy()
        # Create new `Title` feature and create a new numeric feature for each different title:
        X['Title'] = X['Name'].str.extract(pat='([A-Z][a-z]+\.)')
        X['Title'][~X['Title'].isin(['Mr.', 'Miss.', 'Mrs.', 'Master.'])] = 'Misc.'
        X = pd.get_dummies(X, columns=['Title'])
        # Replace `Sex` string entries with 0/1:
        X.replace({'male': 0, 'female': 1}, inplace=True)
        # Categorize `Age` Feature:
        X['AgeCat'] = pd.cut(X['Age'], bins=[0, 5, 12, 18, 25, 35, 60, np.inf], labels=[1, 2, 3, 4, 5, 6, 7]).astype(int)
        # Create new feature `FamilySize`:
        X['FamilySize'] = X['SibSp'] + X['Parch'] + 1  # +1: Person itself
        # Create new feature `GroupSize` (not necessarily relatives):
        X['GroupSize'] = X['Ticket'].map(X['Ticket'].value_counts())
        # Categorize `Fare` Feature:
        X['PersonPerTicket'] = X['Ticket'].map(X['Ticket'].value_counts())
        X['FarePerPerson'] = X['Fare'] / X['PersonPerTicket']
        bins, labels = [-1, 1, 10, 20, 30, 50, np.inf], [1, 2, 3, 4, 5, 6]
        X['FareCat'] = pd.cut(X['FarePerPerson'], bins=bins, labels=labels).astype(int)
        # Drop all non-used features:
        drop = ['Age', 'Name', 'SibSp', 'Parch', 'Ticket', 'PersonPerTicket', 'Fare', 'FarePerPerson']
        X.drop(drop, axis=1, inplace=True)
        return X


class TitanicMinMaxScaler(MinMaxScaler):

    def transform(self, X, **kwargs):
        assert isinstance(X, pd.DataFrame), 'This scaler is designed to work on pandas DataFrames!'
        array_scaled = super().transform(X, **kwargs)  # The normal Scaler returns a numpy array instead of a DataFrame!
        return pd.DataFrame(data=array_scaled, columns=X.columns)


df = df_train.copy()
imputer = TitanicImputer()
transformer = TitanicFeatureTransformer()
scaler = TitanicMinMaxScaler()

df_imputed = imputer.fit_transform(df)
df_transformed = transformer.fit_transform(df_imputed)
df_scaled = scaler.fit_transform(df_transformed)
df_scaled.info()
df_scaled.describe()


# %% [markdown]
# # 4) Applying Machine Learning Algorithms


# %% [markdown]
# ## 4.1) Creating a pipeline

# %%
X = df_train.drop(['Survived'], axis=1)
y = df_train['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f'X_train.shape: {X_train.shape}')
print(f'y_train.shape: {y_train.shape}')

# %%
pipeline = make_pipeline(TitanicImputer(), TitanicFeatureTransformer(), TitanicMinMaxScaler())
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)


# %%
def best_grid_params(grid_search, add_columns=None, sort_by='mean_test_score', ascending=False, n=5):
    columns = ['mean_test_score', 'std_test_score', 'params']
    if add_columns is not None:
        columns.extend(add_columns)
    df_cvres = pd.DataFrame.from_dict(grid_search.cv_results_)[columns]
    df_params = df_cvres['params'].apply(pd.Series)
    df_result = pd.concat((df_cvres.drop(['params'], axis=1), df_params), axis=1)
    df_result['mean_test_score'] = (100*df_result['mean_test_score']).map('{:.2f}%'.format)
    df_result['std_test_score'] = (100*df_result['std_test_score']).map('{:.2f}%'.format)
    return df_result.sort_values(by=sort_by, ascending=ascending).head(n)


# %% [markdown]
# ## 4.2) Algorithm 1: DecisionTree

# ### 4.2.1) Flacher DecisionTree für Intuition
# Zunächst wollen wir mittels eines Baumes mit geringer Tiefe eine erste Intuition entwickeln, was die Entscheidungs-
# grundlagen für die folgenden Algorithmen sein könnten. Decision Trees helfen hierbei, da man sie leicht visualisieren
# kann und man außerdem Informationen über die Features mit der größten Wichtigkeit erhält. Wir wählen zu diesem Zweck
# erstmal eine maximale Tiefe von `max_depth=4`, damit der Plot nicht zu unübersichtlich wird.

# %%
model = DecisionTreeClassifier(max_depth=4, random_state=SEED)
model = model.fit(X_train, y_train)

# %%
try:
    from graphviz import Source
    from IPython.display import Image, display
    graph = Source(export_graphviz(model, out_file=None, feature_names=X_train.columns))
    png_bytes = graph.pipe(format='png')
    with open('dtree.png','wb') as f:
        f.write(png_bytes)
    display(Image(png_bytes))
except ModuleNotFoundError:
    plot_tree(model)
    plt.show()

# %%
print(pd.Series(model.feature_importances_, index=X_train.columns).sort_values(ascending=False))

# %% [markdown]
# Das Feature `Title_Mr.` war für unseren Baum das wichtigeste Entscheidungskriterium. Unserer Meinung nach ist es sogar
# mächtiger als `Sex`, welches in der anfänglichen Korrelationsbetrachtung sehr stark mit der Überlebenschance
# korreliert war. `Title_Mr.` vereinigt Informationen über das Geschlecht, als auch über das Alter in sich und hat von
# daher einen hohen Informationsgehalt. "Master", also Jungen, sind zwar auch männlich, wurden aber viel häufiger
# gerettet als Männer höheren Alters, was das Feature `Title_Mr.` gut trennen kann. Da `Title_Mr.` bereits Informationen
# über das Geschlecht enthält, wird dieses lediglich im `False`-Part nach der `Title_Mr.`-Abfrage benötigt und zwar
# erst im Falle einer positiven Entscheidung für die Kategorie `Title_Misc.`, welche als einziger Titel Frauen und
# Männer enthält.


# %% [markdown]
# ### 4.2.2) Cross Validation mit Decision Tree
# Das Trainingsset wird in 5 Sub-Sets eingeteilt, von denen iterativ eine als Testmenge benutzt wird. Mit den jeweils
# übrigen Daten wird der Algorithmus trainiert. Damit bleiben die eigentlichen Validierungsdaten unberührt und die
# Voraussagefähigkeit des Algorithmus ist besser einzuschätzen als bei Verwendung einer festen Trainings- und
# Validierungsmenge.

# %%
def calculate_cv(model, X_train, y_train, cv=5):
    scores_dict = cross_validate(model, X_train, y_train, cv=cv, return_train_score=True)
    df_score = pd.DataFrame.from_dict(scores_dict)
    mean_series = df_score.mean().rename('mean')
    std_series = df_score.std().rename('std')
    df_score = df_score.append([mean_series, std_series])
    df_score['test_score'] = (100*df_score['test_score']).map('{:.2f}%'.format)
    df_score['train_score'] = (100*df_score['train_score']).map('{:.2f}%'.format)
    return df_score

calculate_cv(model, X_train, y_train)

# %% [markdown]
# Interessanterweise ist schon dieser, doch recht kurze, Baum bereits recht gut darin einen hohen Cross-Validation-Score
# zu erreichen (im Vergleich zu anderen Vergleichen online, die ebenfalls bei 79-84% lagen).
#
# # <img src="kaggle_scores.png">
#
# Aufgrund des recht hohen Scores scheint uns der Decision Tree Classifier ein geeigneter Algorithmus für unser Datenset
# zu sein. Im nächsten Schritt wollen wir daher seine Parameter optimieren.


# %% [markdown]
# ## 4.2.3) Grid Search für verschiedene Parameter im Decision Tree:
# Um die bestmöglichen Parameter für optimale Voraussagefähigkeit des Decision Trees zu ermitteln, verwenden wir einen
# Grid Search. Die Parameter, die wir variieren wollen, sind 'Max Depth' und 'Criterion'. Die Tiefe des Baums sollte
# beschränkt sein um ein Overfitting zu vermeiden. Für die Aufteilung an den Knoten kann man zwischen den Kriterien
# 'Gini' (Inhomogenität der Entstehenden Gruppen) und 'Entropy' (Informationsgewinn durch die Teilung) wählen.

# %%
# Parameter für DesicionTree Gridsearch: unterschiedliche Tiefen und Entscheidungskriterien:
param_grid = {"max_depth":[1, 3, 4, 5, 6, 7, 8, 10, 20, None], "criterion":["gini", "entropy"]}
decision_tree = DecisionTreeClassifier(random_state=SEED)
grid_search = GridSearchCV(decision_tree, param_grid, return_train_score=True, cv=5)
grid_search.fit(X_train, y_train)

print('Beste Parameter-Kombinationen Gridsearch:')
df_decision_tree_results = best_grid_params(grid_search)
df_decision_tree_results

# %% [markdown]
# Von den betrachteten Werten liefern kurze Entscheidungsbäume mit Tiefen von 3 bis 4 die besten (sehr ähnlichen)
# Scores, die Parameter 'Gini' und 'Entropy' liefern keine wesentlichen Unterschiede in den Scores. Auch bei
# unbeschränkter Baumtiefe werden schlechtere Scores erzielt, was auf Overfitting schließen lässt, bei dem zwar die
# Trainings-, nicht aber die Validierungsdaten gut vorhergesagt werden.
#
# Wir hatten nicht erwartet, dass die Scores so kurzer Entscheidungsbäume schon bei 82 % liegen würden, sondern hatten
# eher deutlich niedrigere Werte erwartet. Zu beachten ist jedoch, dass nur der Mittelwert bei 82 % liegt und wir
# Schwankungen von ca. `std_test_score=2.5%` beim "besten" Baum zu verzeichnen haben. Falls andere Algorithmen
# mit ähnlichem Mittelwert kleinere Schwankungen aufweisen, so wären diese dem DecisionTree vorzuziehen.


# %% [markdown]
# ## 4.3) Algorithmus 2: Categorical Naive Bayes
# Wenn man davon ausgeht, dass bestimmte Features die Wahrscheinlichkeit das Schiffsunglück zu überleben erhöhen oder
# erniedrigen, dann kann man auch einen Bayes-Algorithmus anwenden. Den 'Categorical Naive Bayes' halten wir für
# unsere Daten am passendsten, da sie kategorischer Art sind, aber nicht ausschließlich binär.
#
# Allerdings erwarten wir etwas schlechtere Scores, da unsere Features nicht gänzlich von einander unabhängig sind, wie
# es für diesen Algorithmus eigentlich Voraussetzung ist.

# %%
bayes_clf = CategoricalNB()
model = bayes_clf.fit(X_train, y_train)
df_bayes_results = calculate_cv(model, X_train, y_train)
df_bayes_results


# %% [markdown]
# Die Scores für den Categorical Naive Bayes liegen im Mittel bei unter 80 %, also wie wir erwartet hatten, etwas unter
# denen der Decision Trees, wobei die Standardabweichung fast doppelt so hoch ist. `train_score` und `test_score`
# Mittelwerte sind sehr ähnlich, was dafür spricht, dass nicht overfitted wurde.


# %% [markdown]
# ## 4.4) Algorithmus 3: KNeighbors Classifier
# Für nicht kontinuierliche Daten eignet sich auch der 'Kneighbors'-Algorithmus. Mit den ursprünglichen Daten hätten wir
# diesen dennoch nicht benutzen können, da Features mit Strings oder mehreren nicht ordinalen Einträgen sich nicht
# sinnvoll verarbeiten lassen. Da wir unsere Daten aufbereitet haben, so dass in den einzelnen Features entweder nur
# jeweils skalierte Binärwerte oder Ordinalwerte stehen, lassen sich Abstände berechnen. Wir wollen testen, ob der
# Parameter 'Distance' sich sinnvoll anwenden lässt, weil wir uns nicht sicher sind. Wir behalten die Euklidische Norm
# bei, weil die Daten keine spezielle Norm erfordern.

# %%
KN_clf = KNeighborsClassifier(weights='distance')
model = KN_clf.fit(X_train, y_train)
print("Scores mit cross_val: (weights = distance)")
calculate_cv(model, X_train, y_train)

# %%
KN_clf = KNeighborsClassifier(weights='uniform')
model = KN_clf.fit(X_train, y_train)
print("Scores mit cross_val: (weights = uniform)")
calculate_cv(model, X_train, y_train)

# %% [markdown]
# Bei beiden Einstellungen für `weight` ist zu verzeichnen, dass der `train_score` deutlich höher ist als der
# `test_score` (außerhalb der Standardabweichungen). Dies deutet darauf hin, dass der KNearestNeighbor-Algorithmus in
# diesem Fall anfällig für Overfitting ist (für `weight='distance` sogar noch deutlich mehr).
#
# Alleine den Parameter `weights` zu variieren ändert den Cross-Validation-Score nicht besonders. Deshalb wollen wir
# als nächstes auch beim KNearestNeighbors Algorithmus einen GridSearch versuchen. Dazu variieren wir die Anzahl der
# nächsten Nachbarn, den `weights` parameter, so wie oben bereits probiert und wir variieren zusätzlich die Norm
# zwischen euklidischer (`p=2`) und der Manhattan Norm (`p=1`):

# %%
neighbor_range = list(range(1, 21)) + [25, 30, 50, 100]
param_grid = {"n_neighbors": neighbor_range, "p": [1,2], "weights": ["distance", "uniform"]}
KN_clf = KNeighborsClassifier()
grid_search = GridSearchCV(KN_clf, param_grid, return_train_score=True, cv=6)
grid_search.fit(X_train, y_train)

# %%
print('Beste Parameterkombinationen Grid Search:')
df_kneighbors_results = best_grid_params(grid_search, n=10)
df_kneighbors_results

# %% [markdown]
# Der K-Nächste-Nachbarn-Algorithmus mit 13 Nachbarn, Manhattan-Norm und uniformen Gewichten ist hier als Gewinner
# hervorgegangen und ist im Score vergleichbar mit dem besten DecisionTree von zuvor. Allerdings schwankt die
# Vorhersagekraft aufgrund der höheren Standardabweichung (4.3 %) deutlich stärker. Für uns ist der KNearestNeighbor
# deshalb nicht der Algorithmus der Wahl.


# %% [markdown]
# ## 4.5) Ensemble Learning mit Random Forest Classifier
# Zu guter Letzt wollen wir noch einen Ensemble-Classifier ausprobieren und haben uns aufgrund der bisher bereits guten
# Performance des DecisionTrees für den Random Forest entschieden. Zuerst schauen wir uns einen Forest exemplarisch an:

# %%
forest_clf= RandomForestClassifier(n_estimators=100, random_state=SEED)
model = forest_clf.fit(X_train, y_train)
calculate_cv(model, X_train, y_train)

# %% [markdown]
# Unser erster Versuch war nicht schlecht, jedoch ist sicher eine bessere Parameterkonfiguration zu finden, die wir
# erneut mittels CVGridSearch suchen. Hierbei variieren wir die Anzahl der Bäume/Estimatoren, deren maximale Tiefe,
# sowie dem `bootstrap` Parameter, den wir an und aus schalten:

# %%
param_grid = {"n_estimators":[10, 50, 100, 150], "max_depth":[1, 2, 3, 4, 5, 6], "bootstrap":[True, False]}
random_forest_clf = RandomForestClassifier(random_state=SEED)
grid_search = GridSearchCV(random_forest_clf, param_grid, return_train_score=True, cv=5)
grid_search.fit(X_train, y_train)
print(grid_search)
print('Beste Parameter Gridsearch für RandomForestClassifier:')
df_random_forest_results = best_grid_params(grid_search, add_columns=['mean_fit_time', 'mean_score_time', 'mean_train_score'])
df_random_forest_results

# %% [markdown]
# Der beste Random Forest ergab einen `mean_test_score=83.0%`, wobei der `mean_train_score=83.2%` betrug, was bedeutet,
# dass weder over- noch underfitted wurde. Die Standardabweichung von `std_test_score=2.6%` ist auch relativ klein.


# %% [markdown]
# ## 4.6) Finales Model

# %%
final_model = grid_search.best_estimator_

final_train_score = final_model.score(X_train, y_train)
final_test_score = final_model.score(X_test, y_test)

print(f'Final train score: {100*final_train_score:.2f}%')
print(f'Final test  score: {100*final_test_score:.2f}%')


# %% [markdown]
# # 5) Fazit

# %%
df_list = [('Decision Tree', df_decision_tree_results),
           ('K Nearest Neighbours', df_kneighbors_results),
           ('Random Forest', df_random_forest_results),
           ('Categorical Bayes', df_bayes_results)]

entry_list = []
for name, df in df_list:
    if name == 'Categorical Bayes':
        score = df['test_score'].loc['mean']
        std = df['test_score'].loc['std']
    else:
        score = df['mean_test_score'].iloc[0]
        std = df['std_test_score'].iloc[0]
    entry_list.append([name, score, std])

df_cv_plot = pd.DataFrame(entry_list, columns=['Name', 'Mean CV Score [%]', 'Std CV Score [%]'])
df_cv_plot.set_index('Name')
df_cv_plot['Mean CV Score [%]'] = df_cv_plot['Mean CV Score [%]'].str.replace('%', '').astype(float)
df_cv_plot['Std CV Score [%]'] = df_cv_plot['Std CV Score [%]'].str.replace('%', '').astype(float)

final_list = [['Training Data', 100*final_train_score],
              ['Test Data', 100*final_test_score]]
df_final_plot = pd.DataFrame(final_list, columns=['Name', 'Final score [%]'])
df_final_plot.set_index('Name')

fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'width_ratios': [2, 1]})
sns.barplot(data=df_cv_plot, x='Name', y='Mean CV Score [%]', capsize=.2, ax=axes[0])
axes[0].errorbar(x=range(4),
                 y=df_cv_plot['Mean CV Score [%]'].values,
                 yerr=df_cv_plot['Std CV Score [%]'].values,
                 fmt='.k', capsize=10)
axes[0].set_ylim(75, 90)
sns.barplot(data=df_final_plot, x='Name', y='Final score [%]', palette='Paired', ax=axes[1])
axes[1].set_ylim(75, 90)

# %% [markdown]
# Wurden Frauen und Kinder zuerst gerettet?
# Folgendes konnte bereits anhand eines Blicks auf die vorliegenden Datensätze bereits festgestellt werden: Von 577
# männlichen Passagieren sind 486 gestorben (84 %) und von den 314 weiblichen Passagieren nur 81 (25 %). Somit ist
# wahrscheinlich, dass das Geschlecht zumindest einen großen Einfluss auf die Überlebenschance hatte.
#
# Das Datenset beinhaltet die Datensätze von 891 der über 1300 Passagiere, die eine Überfahrt auf der Titanic gebucht
# hatten, also ungefähr 68 %. Somit sind die Aussagen, die das Datenset liefern kann, begrenzt. Eine detaillierte
# Untersuchung der Daten sollte Aufschluss darüber geben, welche Faktoren für das Überleben noch eine Rolle gespielt
# haben. Bereits bei einem kurzen Überblick mit einem kurzen Entscheidungsbaum der Tiefe 4 wurde klar, dass das Feature
# ‚Titel Mr.‘, welches die Eigenschaften ‚männlich‘ und ‚älter als 18‘ vereint, einen entscheidenden Einfluss auf die
# Überlebenschance hat.
# Insgesamt erhalten wir auch in der Menge der getesteten Algorithmen mit den Entscheidungsbäumen des Random Forest die
# effizientesten Vorhersagen mit dem besten Score und der besten Standardabweichung. Auch liefert er ähnliche Werte für
# Trainings- und Testdaten. Verglichen wurden: Decision Tree, Random Forest, Categorical Naive Bayes und KNeigbors.
# Diese bieten sich am ehesten für die Untersuchung unserer kategorischen Daten an.


# %% [markdown]
# # 6) Lessons learned

# %% [markdown]
# ## 6.1) Was wir mitnehmen aus dem Projekt
#
# - Es lohnt sich, ausführlich in die Daten zu schauen und ggf. neue Kategorien zu bilden, weil man dadurch
#
#   - die Weiterverarbeitung sehr vereinfachen
#
#   - die Algorithmen verkürzen
#
#   - und die Voraussagen optimieren kann.
#
# - Die reine Datenvorverarbeitung kostet bereits genauso viel oder sogar mehr Zeit, als die Durchführung des ML und die
#   Bewertung der Algorithmen und ihrer Vorhersagefähigkeit.
#
# - Es gibt Abstände zwischen kategorialen Daten, aber man sollte sich bewusst machen, wie die Algorithmen diese
#   verwerten. Beispielsweise könnte der KneighborsClassifier Features übergewichten, die nach dem Skalieren nur Nullen
#   und Einsen enthalten gegenüber Features, die Werte zwischen Null und Eins haben. Unsere ausführlichen Diskussionen
#   haben sehr zum Verstehen der Daten beigetragen.
#
# - Man darf nicht vergessen, dass die Testdaten NaNs an anderen Stellen enthalten können, als die Trainingsdaten. Daher
#   sollte man sich vorher überlegen (und auch testen), wie man damit umgeht.
#
# - Es ist aus 2 Gründen absolut sinnvoll, die Dokumentation gleich ausführlich zum Code dazuzuschreiben:
#
#   1.  Weil man schon am nächsten Tag nicht mehr unbedingt weiß, warum man einen bestimmten Parameter getestet hat.
#       Häufig liest man sich kurz ein und dann ist es hilfreich, das Fazit notiert zu haben.
#
#   2.  Die Dokumentation kostet viel Zeit. Verschiebt man sie aufs Projektende, ist die Zeit zu knapp.
#
# - Für die Dokumentation ist es sehr hilfreich ein Programm zu benutzen, dass Kommentare und Code-Teile in pdfs
#   überführen kann.
#
# - Es ist nicht notwendig, aber sehr hilfreich, wenn alle mit demselben Programm arbeiten (Python-Version,
#   Interpreter). VSCode eignet sich leider besser als PyCharm (leider, weil es nicht bei allen funktioniert hat).
#
# - Eine gute Programmierumgebung ist extrem hilfreich, da sie enorm viel Zeit spart.
#
# - Coden ist nicht das eigentliche Lernziel, sondern Machine Learning ist ein spannender Anlass um sich Coden anhand
#   von konreten Projekten beizubringen.
#
# - Ein Projekt mit mehreren Personen ist eigentlich immer interessant, weil so viele verschiedene Blickwinkel, Ideen
#   und Vorkenntnisse zusammenkommen.
#
# - Pandas ist ein sehr mächtiges Paket zur Manipulation von Datensätzen und zur Feature Generierung, aber auch zur
#   übersichtlichen Darstellung von Daten. Ohne viel Erfahrung ist die Lernschwelle jedoch sehr hoch.
#
# - Obwohl man auch Matplotlib benutzen könnte ist Seaborn für statistische Plots vorzuziehen, vor allem, da man sehr
#   viel Zeit sparen kann, wenn man die Integration von Panda Datenframes nutzt, ohne sich um die Formatierung kümmern
#   zu müssen.
#
# - Denken in mehr als 3 Dimensionen ist nicht immer einfach...
#
# - Sabine:
#
#   - Ich persönlich glaube, dass ich mehr vom Kurs und vom Projekt profitiert hätte, wenn ich schon bessere
#     Vorkenntnisse gehabt hätte. Viele Informationen sind nicht hängen geblieben, weil zu viel gänzlich neu war.
#
#   - Schriftliche, exakt formulierte Aufgabenstellungen sind extrem hilfreich, wenn man eh unsicher ist.
#
#   - Zum Kurs: Für mich als Anfänger im Coden wäre es hilfreicher gewesen, kleine Programmier-Aufgaben (wirklich kleine)
#     zu Anfang zu bekommen (und wenn es nur 1 – 2 pro Tag sind), deren Lösungen am nächsten Tag auch besprochen werden
#     (Vormachen oder Musterlösung), ggf. von der Hauptgruppe getrennt nur für die, die es brauchen.


# %% [markdown]
# ## 6.2) Was wir mit mehr Zeit noch hätten machen können
#
# - Die starken Schwankungen beim Cross Validation Score könnten eventuell mit der Aufteilung des K-Fold Split (hier
#   `K=5`) zu tun haben, falls die Passagierliste eine gewisse Vorsortierung besaß. Es könnte sich lohnen, einen
#   zuvor einen ShuffleSplit laufen zu lassen um diesen Effekt potentiell zu reduzieren.
#
# - Einen Boosting-Algorithmus auszuprobieren wäre noch interessant gewesen.
#
# - Neuronale Netze können prinzipiell jede Funktion approximieren und wären deswegen auch ein Kandidat gewesen, den man
#   noch hätte testen können.
#
# - Eventuell wäre eine PCA interessant gewesen um zu sehen, ob es überflüssige Features gibt, die man weglassen kann.
#
# - Wir haben die Features `Cabin` und `Embarked` verworfen (aus Gründen der Unvollständigkeit und der Intuition, dass
#   der Starthafen keinen Einfluss haben sollte), mit mehr Zeit hätte man aus diesen Daten eventuell aber auch noch
#   Informationen ziehen können (beispielsweise das `Deck` aus der Kabinennummer).

# %%
