TRACK_WORDS = ['Earthquake']
TABLE_NAME = "Earthquake"
TABLE_ATTRIBUTES = "id_str VARCHAR(255), created_at DATETIME, text VARCHAR(255), \
            polarity INT, subjectivity INT, user_created_at VARCHAR(255), user_location VARCHAR(255), \
            longitude DOUBLE, latitude DOUBLE"


TABLE_NAME_TWO = "TrainingSet"
TABLE_ATTRIBUTES_TWO = "id_str VARCHAR(255), keyword VARCHAR(255), location VARCHAR(255), Tweet VARCHAR(255), target DOUBLE DEFAULT NULL"
