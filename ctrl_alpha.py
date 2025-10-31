import pandas as pd
import numpy as np
import pandas_ta as ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")

class CtrlAlpha:
    
    def __init__(self):
        # Final Hyperparameters
        self.h = 20
        self.upper_threshold = 0.003
        self.lower_threshold = -0.003
        self.execution_delay = 1
    
        
        self.model = RandomForestClassifier(
            n_estimators=150, max_depth=10, min_samples_leaf=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
        self.scaler = StandardScaler()

       
        self.history_df = pd.DataFrame()
        self.features_list = None
        self.min_history_size = 50

    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if not isinstance(df_copy.index, pd.DatetimeIndex):
            df_copy.index = pd.to_datetime(df_copy.index, unit='s')

        # Adding technical indicators
        df_copy.ta.rsi(append=True)
        df_copy.ta.macd(append=True)
        df_copy.ta.bbands(append=True)
        df_copy.ta.obv(append=True)

        df_copy.fillna(method='ffill', inplace=True)
        return df_copy

    def _create_labels(self, df: pd.DataFrame) -> pd.Series:
        forward_returns = (df['close'].shift(-self.h) / df['close']) - 1
        conditions = [
            (forward_returns > self.upper_threshold),
            (forward_returns < self.lower_threshold),
        ]
        choices = [1, -1]
        labels = np.select(conditions, choices, default=0)
        labels = pd.Series(labels, index=df.index)
        labels.iloc[-self.h:] = np.nan
        return labels

    def train(self, train_df: pd.DataFrame):
        df = train_df.copy()
        df.columns = [col.lower() for col in df.columns]

        df_features = self._create_features(df)
        df_labels = self._create_labels(df_features)

        df_combined = df_features.join(df_labels.rename('signal'))
        df_combined.dropna(inplace=True)

        X = df_combined.drop(columns=['signal'])
        y = df_combined['signal']

        if len(X) < self.min_history_size:
            return

        self.features_list = [col for col in X.columns if col in df_features.columns]
        X = X[self.features_list]

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self.history_df = df.tail(self.min_history_size * 2)

    def predict(self, row: pd.Series, timestamp: int) -> int:
        df = row.to_frame().T
        df.columns = [col.lower() for col in df.columns]
        
        _ = timestamp # We accept the timestamp as we do in the real world, but don't use it in this logic

        self.history_df = pd.concat([self.history_df, df], ignore_index=True)
        
        if len(self.history_df) < self.min_history_size or self.features_list is None:
            return 0

        features_df = self._create_features(self.history_df.copy())
        latest_features = features_df[self.features_list].iloc[-self.execution_delay]

        if latest_features.isnull().any():
            return 0
            
        scaled_features = self.scaler.transform(latest_features.values.reshape(1, -1))
        prediction = self.model.predict(scaled_features)
        
        return int(prediction[0])
