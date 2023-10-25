import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


rest=pd.read_csv("resturants_data.csv")
rest.food_type1.fillna(rest.food_type, inplace=True)
rest.dropna(subset=['food_type1'], inplace=True)


rest=rest.drop(['address_line1','address_line2','food_type','location','number_of_reviews','opening_hour',
                   'out_of','phone','price_range'],axis=1) 
rest=rest.drop_duplicates(subset='restaurant_name', keep="last")
rest['restaurant_name'] = rest['restaurant_name'].str.strip()

rest = rest.reset_index()


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(rest['city'])
rest['city'] = le.transform(rest['city'])

le.fit(rest['food_type1'])
rest['food_type1'] = le.transform(rest['food_type1'])



def calculate_similarity_matrix(rest):
    attributes = rest[['city', 'food_type1', 'review_score']].values
    return cosine_similarity(attributes, attributes)

similarity_matrix = calculate_similarity_matrix(rest)

def recommend_restaurants(restaurant_name, similarity_matrix, num_recommendations=5):
    restaurant_index = rest[rest.restaurant_name == restaurant_name].index[0]
    similar_restaurants = list(enumerate(similarity_matrix[restaurant_index]))
    similar_restaurants = sorted(similar_restaurants, key=lambda x: x[1], reverse=True)
    similar_restaurants = similar_restaurants[1:num_recommendations + 1]  
    recommended_restaurants = [rest.iloc[i[0]]['restaurant_name'] for i in similar_restaurants]
    return recommended_restaurants




if __name__=="__main__":
    def calculate_similarity_matrix(rest):
        attributes = rest[['city', 'food_type1', 'review_score']].values
        return cosine_similarity(attributes, attributes)

    similarity_matrix = calculate_similarity_matrix(rest)

    def recommend_restaurants(restaurant_name, similarity_matrix, num_recommendations=5):
        restaurant_index = rest[rest.restaurant_name == restaurant_name].index[0]
        similar_restaurants = list(enumerate(similarity_matrix[restaurant_index]))
        similar_restaurants = sorted(similar_restaurants, key=lambda x: x[1], reverse=True)
        similar_restaurants = similar_restaurants[1:num_recommendations + 1]  
        recommended_restaurants = [rest.iloc[i[0]]['restaurant_name'] for i in similar_restaurants]
        return recommended_restaurants
    
    inp = input("Enter your favorite resturant name:")
    print(recommend_restaurants(inp, similarity_matrix))
