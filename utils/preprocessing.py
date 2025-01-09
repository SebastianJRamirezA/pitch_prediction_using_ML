import pandas as pd

def filter_pitch_types(data, valid_pitch_dict):
    return data[data['pitch_type'].str.contains('|'.join(list(valid_pitch_dict.keys())), na=False)]

def filter_regular_season(data):
    return data[data['game_type'] == 'R']

def get_repertoire(data):
    repertoire_abb = data['pitch_type'].unique()
    repertoire_full = data['pitch_name'].unique()
    return repertoire_abb, repertoire_full

def calculate_top_pitch(data, valid_pitch_dict):
    top_pitch = data['pitch_type'].value_counts().head(1)
    top_pitch_name = valid_pitch_dict[top_pitch.index[0]]
    top_pitch_freq = int((top_pitch.values[0] / len(data)) * 100)
    print(f'{top_pitch_name} is thrown {top_pitch_freq}% of the time')

def add_plate_app_id(data):
    data['plate_app_id'] = data['game_pk'].astype(str) + data['batter'].astype(str) + data['at_bat_number'].astype(str)
    return data

def sort_data(data):
    return data.sort_values(['game_date', 'game_pk', 'plate_app_id', 'pitch_number'], ascending=True)

def score_diff(row):
    if row['inning_topbot'] == 'Top':
        return row['home_score'] - row['away_score']
    else:
        return row['away_score'] - row['home_score']

def engineer_features(data):
    data['previous_pitch'] = data['pitch_type'].shift(1)
    data.loc[data['pitch_number'] == 1, 'previous_pitch'] = None

    data['previous_zone'] = data['zone'].shift(1)
    data.loc[data['pitch_number'] == 1, 'previous_zone'] = None

    on_base_cols = ['on_3b', 'on_2b', 'on_1b']
    for col in on_base_cols:
        data[col] = data[col].fillna(0).astype(int)
        data.loc[data[col] != 0, col] = 1

    data['score_diff'] = data.apply(score_diff, axis=1)
    return data

def select_features(data):
    selected_features = [
        'plate_app_id', 'previous_pitch', 'previous_zone', 'pitch_number',
        'inning', 'on_3b', 'on_2b', 'on_1b', 'score_diff', 'balls', 'outs_when_up', 'pitch_type', 'zone'
    ]
    return data[selected_features]

def preprocess_data(file_path):
    # Define valid pitch types
    valid_pitch_dict = {
        'FF': 'Four-Seam Fastball', 'FT': 'Two-Seam Fastball',
        'CH': 'Change-up', 'CU': 'Curveball',
        'FC': 'Cutter', 'EP': 'Eephus',
        'FO': 'Forkball', 'KC': 'Knuckle Curve',
        'KN': 'Knuckleball', 'SC': 'Screwball',
        'SI': 'Sinker', 'SL': 'Slider',
        'FS': 'Splitter',
    }

    data = pd.read_csv(file_path)
    data = filter_pitch_types(data, valid_pitch_dict)
    data = filter_regular_season(data)
    repertoire_abb, repertoire_full = get_repertoire(data)
    # calculate_top_pitch(data, valid_pitch_dict)
    data = add_plate_app_id(data)
    data = sort_data(data)
    data = engineer_features(data)
    data = select_features(data)
    data = pd.get_dummies(data, columns=['previous_zone', 'previous_pitch', 'inning'], dtype=int)

    # data.to_csv(output_path, index=False)
    return data, repertoire_abb

