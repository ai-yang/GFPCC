# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import pandas as pd
pd.set_option('display.max_columns', 20)
movie = pd.read_csv("/kaggle/input/movielens-20m-dataset/movie.csv")
rating = pd.read_csv("/kaggle/input/movielens-20m-dataset/rating.csv")

df = movie.merge(rating, how="left", on="movieId")
df.head()
# Let's get to the movies with less than 1000 reviews:
comment_counts = pd.DataFrame(df["title"].value_counts())
rare_movies = comment_counts[comment_counts["title"] <= 1000].index

# Let's get access to movies with over 1000 reviews:
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.shape

# Let's create the User Movie Df:
user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

# There are 3159 movies that 138493 users have voted for.
user_movie_df.shape

# item-based movie recommendation example:
movie_name = "Matrix, The (1999)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

# Let's determine the movies that the user watched.

# Let's choose random user:
# random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)
random_user = 28491