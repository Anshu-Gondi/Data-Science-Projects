import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path):
    """Load dataset from a CSV file"""
    return pd.read_csv(file_path)

def preprocess_data(df):
    """Handle missing values, drop unnecessary columns, categorize car make, and perform label encoding"""

    # Drop unnecessary columns
    df.drop(columns=['model', 'segment'], inplace=True)

    # Drop rows with missing values
    df.dropna(inplace=True)

    # Categorizing the car make
    def car_make(make):
        if make in ['mazda', 'mg', 'rover', 'alfa-romeo', 'audi', 'peugeot', 'chrysler', 'bmw', 'aston-martin', 'jaguar', 'land-rover']:
            return 'Luxury European'
        elif make in ['renault', 'dacia', 'citroen', 'volvo', 'fiat', 'opel', 'seat', 'volkswagen', 'citroen', 'skoda', 'mini', 'smart']:
            return 'Mainstream European'
        elif make in ['gaz', 'aro', 'lada-vaz', 'izh', 'raf', 'bogdan', 'moskvich', 'uaz', 'luaz', 'wartburg', 'trabant', 'proton', 'fso', 'jac', 'iran-khodro', 'zotye', 'tagaz', 'saipa', 'brilliance']:
            return 'Russian/Eastern European'
        elif make in ['toyota', 'nissan', 'asia', 'mitsubishi', 'chery', 'hyundai', 'honda', 'ssangyong', 'suzuki', 'daihatsu', 'kia', 'changan', 'lexus', 'isuzu', 'great-wall', 'daewoo', 'vortex', 'infiniti', 'byd', 'geely', 'haval', 'acura', 'scion', 'tata', 'datsun', 'ravon', 'proton', 'jac']:
            return 'Asian'
        elif make in ['oldsmobile', 'gmc', 'chrysler', 'plymouth', 'ford', 'cadillac', 'jeep', 'mercury', 'lincoln', 'buick', 'saturn', 'pontiac', 'chevrolet']:
            return 'American'
        elif make in ['porsche', 'bentley', 'maserati', 'tesla', 'mclaren']:
            return 'Specialty'
        else:
            return 'Other'

    df['make_segment'] = df['make'].apply(car_make)

    # Drop 'make' column as it is no longer needed
    df.drop(columns=['make'], inplace=True)

    # Define columns to encode
    cols = ['condition', 'fuel_type', 'transmission',
            'color', 'drive_unit', 'make_segment']

    # Label Encoding
    le = LabelEncoder()
    for col in cols:
        df[col] = le.fit_transform(df[col])

    return df

if __name__ == "__main__":
    df = load_data("data/cars.csv")
    df = preprocess_data(df)

    # Save preprocessed data (before outlier removal)
    df.to_csv("data/preprocessed_data.csv", index=False)

    print("✅ Data preprocessing complete! Saved as 'data/preprocessed_data.csv'.")
    print(df.head())
