def test_dummy():
    assert True

def test_my_def():
    import batch
    assert batch.my_def() == 1

def test_prepare_data():
    from datetime import datetime
    import pandas as pd
    import batch

    def dt(hour, minute, second=0):
        return datetime(2023, 1, 1, hour, minute, second)
    
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    df_prepared = batch.prepare_data(df, categorical)

    columns_expected = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'duration']
    data_expected = [
        (None, None, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0)  
    ]

    df_expected = pd.DataFrame(data_expected, columns=columns_expected)
    df_expected[categorical] = df_expected[categorical].fillna(-1).astype('int').astype('str')

    pd.testing.assert_frame_equal(df_prepared.reset_index(drop=True), df_expected.reset_index(drop=True))




