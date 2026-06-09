from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.model_trainer import ModelTrainer

loader = DataLoader("data/Churn_Modelling.csv")
df = loader.load_data()

processor = Preprocessor()
X, y = processor.preprocess(df)

trainer = ModelTrainer()
trainer.train(X, y)